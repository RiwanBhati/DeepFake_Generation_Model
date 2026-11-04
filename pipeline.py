"""
pipeline.py
===========
Master End-to-End Orchestrator for Hinglish DeepFake Generation & LLM Ablation Study.

Workflow:
1. Extract Reference Audio from Input Video (24kHz Mono WAV via ffmpeg).
2. Refine Target Hinglish Text using LLM (Hindi in Devanagari, English in Latin script, natural pauses).
3. Synthesize Cloned Hinglish Audio via IndicF5 (Produces: Before-LLM & After-LLM WAVs).
4. Apply Real-Time Lip-Sync via MuseTalk (Produces: Before-LLM & After-LLM MP4s).
5. Record metadata into data/metadata.csv for evaluation & report comparison.
"""

import os
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.extract_audio import extract_audio
from src.features.llm_refiner import refine_target_text
from src.models.generate_tts import generate_tts_pair
from src.models.lipsync_musetalk import run_musetalk_pair

METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"


def log_experiment_metadata(row_data: dict):
    """Logs the experiment run into a central metadata CSV file."""
    file_exists = METADATA_FILE.exists()
    headers = [
        "timestamp", "sample_id", "source_video", "reference_audio",
        "reference_text", "raw_target_text", "refined_target_text",
        "raw_audio_path", "refined_audio_path", "video_before_llm", "video_after_llm"
    ]

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"[METADATA] Experiment logged in: {METADATA_FILE}")


def run_pipeline(
    video_path: str,
    reference_text: str,
    target_text: str,
    refined_text: str = None,
    sample_id: str = "sample_01",
    skip_video_sync: bool = False,
    bbox_shift: int = 0
):
    print("="*65)
    print(f"🚀 STARTING DEEPFAKE HINGLISH PIPELINE: [{sample_id}]")
    print("="*65)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"\n[ERROR] Source video file not found at: {video_path}")
        print("[ACTION REQUIRED] Please place your video inside:")
        print(f"       -> {PROJECT_ROOT / 'data' / 'raw_videos'}")
        print("       Then re-run the pipeline.\n")
        return

    # -------------------------------------------------------------
    # Step 1: Extract Audio from Video
    # -------------------------------------------------------------
    print("\n[STEP 1/4] Extracting Reference Audio from Video...")
    ref_audio_path = PROJECT_ROOT / "data" / "reference_data" / f"{sample_id}_ref.wav"
    extracted = extract_audio(
        video_path=str(video_path),
        output_audio_path=str(ref_audio_path),
        sample_rate=24000
    )
    if not extracted:
        print("[ERROR] Failed to extract reference audio. Exiting.")
        return

    # -------------------------------------------------------------
    # Step 2: LLM Text Refinement
    # -------------------------------------------------------------
    print("\n[STEP 2/4] Refining Hinglish Text with LLM...")
    if refined_text:
        raw_text = target_text
        refined_text = refined_text
    else:
        refinement = refine_target_text(target_text)
        raw_text = refinement["raw_text"]
        refined_text = refinement["refined_text"]

    print("\n  • BEFORE LLM (Raw Input):")
    print(f"    \"{raw_text}\"")
    print("\n  • AFTER LLM (Devanagari Hindi + English Script + Punctuation):")
    print(f"    \"{refined_text}\"")

    # -------------------------------------------------------------
    # Step 3: IndicF5-Hinglish Voice Cloning
    # -------------------------------------------------------------
    print("\n[STEP 3/4] Generating Hinglish Speech (IndicF5 Flow Matching)...")
    out_dir_raw = PROJECT_ROOT / "data" / "generated_audio" / "before_llm"
    out_dir_refined = PROJECT_ROOT / "data" / "generated_audio" / "after_llm"

    try:
        tts_results = generate_tts_pair(
            ref_audio_path=str(ref_audio_path),
            ref_text=reference_text,
            raw_target_text=raw_text,
            refined_target_text=refined_text,
            output_dir_raw=str(out_dir_raw),
            output_dir_refined=str(out_dir_refined),
            sample_name=sample_id,
            nfe_steps=32
        )
    except Exception as e:
        print(f"[WARNING] Local TTS generation failed or dependencies missing: {e}")
        print("[INFO] If you are running on CPU, you can trigger this script inside the GPU Pod.")
        tts_results = {
            "raw_audio_path": str(out_dir_raw / f"{sample_id}.wav"),
            "refined_audio_path": str(out_dir_refined / f"{sample_id}.wav")
        }

    # -------------------------------------------------------------
    # Step 4: MuseTalk Lip-Sync Video Generation
    # -------------------------------------------------------------
    video_outputs = {"video_before_llm": "Skipped", "video_after_llm": "Skipped"}
    if not skip_video_sync:
        print("\n[STEP 4/4] Generating Lip-Synced Deepfake Videos (MuseTalk)...")
        try:
            video_outputs = run_musetalk_pair(
                video_path=str(video_path),
                raw_audio_path=tts_results["raw_audio_path"],
                refined_audio_path=tts_results["refined_audio_path"],
                sample_id=sample_id,
                bbox_shift=bbox_shift
            )
        except Exception as e:
            print(f"[WARNING] MuseTalk step skipped/error: {e}")
    else:
        print("\n[STEP 4/4] Video lip-sync step skipped as requested.")

    # -------------------------------------------------------------
    # Log Metadata
    # -------------------------------------------------------------
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "sample_id": sample_id,
        "source_video": str(video_path),
        "reference_audio": str(ref_audio_path),
        "reference_text": reference_text,
        "raw_target_text": raw_text,
        "refined_target_text": refined_text,
        "raw_audio_path": tts_results.get("raw_audio_path", ""),
        "refined_audio_path": tts_results.get("refined_audio_path", ""),
        "video_before_llm": video_outputs.get("video_before_llm", ""),
        "video_after_llm": video_outputs.get("video_after_llm", "")
    }
    log_experiment_metadata(log_data)

    print("\n" + "="*65)
    print("🎉 PIPELINE RUN COMPLETE!")
    print("="*65)
    print("Summary of Generated Assets:")
    print(f"1. Ref Audio:         {ref_audio_path}")
    print(f"2. Audio Before LLM:  {tts_results.get('raw_audio_path')}")
    print(f"3. Audio After LLM:   {tts_results.get('refined_audio_path')}")
    print(f"4. Video Before LLM:  {video_outputs.get('video_before_llm')}")
    print(f"5. Video After LLM:   {video_outputs.get('video_after_llm')}")
    print(f"6. Metadata Log:      {METADATA_FILE}")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end Hinglish DeepFake Generator.")
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw_videos" / "input_video.mp4"),
        help="Path to your 10-second input video"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        required=True,
        help="Transcript of what the person is saying in the input video"
    )
    parser.add_argument(
        "--target_text",
        type=str,
        required=True,
        help="The new Hinglish sentence you want to synthesize"
    )
    parser.add_argument(
        "--refined_text",
        type=str,
        default=None,
        help="Optional explicit refined text (English + Devanagari) if target_text is raw Roman Hinglish"
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="sample_01",
        help="Identifier name for this sample"
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Only generate audio, skip MuseTalk lip sync"
    )
    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=0,
        help="MuseTalk bounding box shift value (default: 0 for calibrated canonical lip sync)"
    )

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        reference_text=args.ref_text,
        target_text=args.target_text,
        refined_text=args.refined_text,
        sample_id=args.sample_id,
        skip_video_sync=args.skip_video,
        bbox_shift=args.bbox_shift
    )


if __name__ == "__main__":
    main()

