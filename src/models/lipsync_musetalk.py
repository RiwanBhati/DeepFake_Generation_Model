"""
lipsync_musetalk.py
-------------------
Step 4 of the DeepFake Generation Pipeline:
Runs the MuseTalk real-time lip-synchronization model.
Takes the original source video + the generated Hinglish audio
and produces deepfake lip-synced videos for BOTH Before-LLM and After-LLM audio tracks.
"""

import os
import sys
import yaml
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUSETALK_DIR = PROJECT_ROOT / "models" / "MuseTalk"


def run_musetalk_inference(
    video_path: str,
    audio_path: str,
    output_video_path: str,
    bbox_shift: int = 4
) -> bool:
    """
    Executes MuseTalk lip-sync inference for a single video + audio pair.
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_video_path = Path(output_video_path).resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[ERROR] Source video not found: {video_path}")
        return False
    if not audio_path.exists():
        print(f"[ERROR] Audio track not found: {audio_path}")
        return False

    if not MUSETALK_DIR.exists():
        print(f"[ERROR] MuseTalk repository not found at {MUSETALK_DIR}")
        return False

    # Create temporary inference config for MuseTalk
    config_dict = {
        "task_0": {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "bbox_shift": bbox_shift
        }
    }

    temp_config_path = MUSETALK_DIR / "configs" / "inference" / "current_task.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_config_path, "w") as f:
        yaml.dump(config_dict, f)

    cmd = [
        sys.executable,
        "-m", "scripts.inference",
        "--version", "v1",
        "--unet_config", "./models/musetalk/musetalk.json",
        "--unet_model_path", "./models/musetalk/pytorch_model.bin",
        "--whisper_dir", "./models/whisper",
        "--vae_type", "sd-vae",
        "--inference_config", str(temp_config_path),
        "--bbox_shift", str(bbox_shift),
        "--result_dir", str(output_video_path.parent),
        "--output_vid_name", output_video_path.name,
        "--use_float16"
    ]

    print(f"[INFO] Running MuseTalk inference on: {video_path.name} + {audio_path.name}")
    env = os.environ.copy()
    env["PATH"] = f"/workspace/.local/bin:{os.environ.get('PATH', '')}"
    env["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
    env["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
    env["MPLCONFIGDIR"] = "/tmp"

    try:
        subprocess.run(
            cmd,
            cwd=str(MUSETALK_DIR),
            check=True,
            env=env
        )
        v1_path = output_video_path.parent / "v1" / output_video_path.name
        if v1_path.exists():
            shutil.copy2(str(v1_path), str(output_video_path))
        print(f"[SUCCESS] Deepfake video saved to: {output_video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MuseTalk execution failed: {e}")
        return False


def run_musetalk_pair(
    video_path: str,
    raw_audio_path: str,
    refined_audio_path: str,
    sample_id: str = "sample_01",
    bbox_shift: int = 4
) -> dict:
    """
    Runs MuseTalk for both Before-LLM and After-LLM audio outputs.
    """
    out_video_raw = PROJECT_ROOT / "data" / "generated_videos" / "before_llm" / f"{sample_id}.mp4"
    out_video_refined = PROJECT_ROOT / "data" / "generated_videos" / "after_llm" / f"{sample_id}.mp4"

    print("\n" + "="*50)
    print("STEP 4: MUSETALK LIP-SYNC GENERATION")
    print(f"       (BBox Shift: {bbox_shift} for precise mouth openness & sharp facial hair)")
    print("="*50)

    print("\n[1/2] Generating Video 1 (Before LLM)...")
    run_musetalk_inference(video_path, raw_audio_path, out_video_raw, bbox_shift=bbox_shift)

    print("\n[2/2] Generating Video 2 (After LLM)...")
    run_musetalk_inference(video_path, refined_audio_path, out_video_refined, bbox_shift=bbox_shift)

    return {
        "video_before_llm": str(out_video_raw),
        "video_after_llm": str(out_video_refined)
    }


def main():
    parser = argparse.ArgumentParser(description="Run MuseTalk lip sync for Before and After LLM audios.")
    parser.add_argument("--video", required=True, help="Input source video (.mp4)")
    parser.add_argument("--raw_audio", required=True, help="Path to Before-LLM audio (.wav)")
    parser.add_argument("--refined_audio", required=True, help="Path to After-LLM audio (.wav)")
    parser.add_argument("--sample_id", default="sample_01", help="Sample identifier")

    args = parser.parse_args()
    run_musetalk_pair(
        video_path=args.video,
        raw_audio_path=args.raw_audio,
        refined_audio_path=args.refined_audio,
        sample_id=args.sample_id
    )


if __name__ == "__main__":
    main()
