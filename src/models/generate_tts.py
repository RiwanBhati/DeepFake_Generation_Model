"""
generate_tts.py
---------------
Step 3 of the DeepFake Generation Pipeline:
Generates cloned Hinglish speech using the IndicF5-Hinglish model.
Generates BOTH Before-LLM and After-LLM audio outputs to enable comparative analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import soundfile as sf

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.indicf5_model import IndicF5Hinglish


def generate_tts_pair(
    ref_audio_path: str,
    ref_text: str,
    raw_target_text: str,
    refined_target_text: str,
    output_dir_raw: str,
    output_dir_refined: str,
    sample_name: str = "sample_01",
    nfe_steps: int = 32,
    model_name_or_path: str = "Saravananravi/indicf5-hinglish"
) -> dict:
    """
    Synthesizes both Before-LLM and After-LLM audio tracks using IndicF5-Hinglish.
    """
    ref_audio_path = Path(ref_audio_path)
    output_dir_raw = Path(output_dir_raw)
    output_dir_refined = Path(output_dir_refined)

    # 1. Validation
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found at: {ref_audio_path}")
    if not ref_text or not ref_text.strip():
        raise ValueError("Reference text is required for voice cloning!")

    output_dir_raw.mkdir(parents=True, exist_ok=True)
    output_dir_refined.mkdir(parents=True, exist_ok=True)

    raw_wav_path = output_dir_raw / f"{sample_name}.wav"
    refined_wav_path = output_dir_refined / f"{sample_name}.wav"

    # 2. Load model
    print(f"\n[INFO] Initializing IndicF5-Hinglish model...")
    model = IndicF5Hinglish.from_pretrained(model_name_or_path)

    # 3. Generate Audio 1: Before LLM (Raw)
    print(f"\n[1/2] Generating BEFORE-LLM Audio...")
    print(f"      Text: \"{raw_target_text}\"")
    audio_raw, sr_raw = model.generate(
        text=raw_target_text,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        speed=1.0,
        nfe_step=nfe_steps
    )
    sf.write(str(raw_wav_path), audio_raw, sr_raw)
    raw_duration = len(audio_raw) / sr_raw
    print(f"      [SAVED] {raw_wav_path} (Duration: {raw_duration:.2f}s)")

    # 4. Generate Audio 2: After LLM (Refined)
    print(f"\n[2/2] Generating AFTER-LLM Audio...")
    print(f"      Text: \"{refined_target_text}\"")
    audio_refined, sr_refined = model.generate(
        text=refined_target_text,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        speed=1.0,
        nfe_step=nfe_steps
    )
    sf.write(str(refined_wav_path), audio_refined, sr_refined)
    refined_duration = len(audio_refined) / sr_refined
    print(f"      [SAVED] {refined_wav_path} (Duration: {refined_duration:.2f}s)")

    return {
        "sample_name": sample_name,
        "raw_audio_path": str(raw_wav_path),
        "raw_duration": raw_duration,
        "refined_audio_path": str(refined_wav_path),
        "refined_duration": refined_duration,
        "sample_rate": sr_raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate IndicF5-Hinglish audio for Before & After LLM.")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio (.wav)")
    parser.add_argument("--ref_text", type=str, required=True, help="Transcript of reference audio")
    parser.add_argument("--raw_text", type=str, required=True, help="Raw target text (Before LLM)")
    parser.add_argument("--refined_text", type=str, required=True, help="Refined target text (After LLM)")
    parser.add_argument("--sample_id", type=str, default="sample_01", help="Sample identifier name")
    parser.add_argument("--nfe", type=int, default=32, help="Diffusion NFE steps (default: 32)")

    args = parser.parse_args()

    out_raw = PROJECT_ROOT / "data" / "generated_audio" / "before_llm"
    out_refined = PROJECT_ROOT / "data" / "generated_audio" / "after_llm"

    results = generate_tts_pair(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        raw_target_text=args.raw_text,
        refined_target_text=args.refined_text,
        output_dir_raw=str(out_raw),
        output_dir_refined=str(out_refined),
        sample_name=args.sample_id,
        nfe_steps=args.nfe
    )

    print("\n" + "="*50)
    print("TTS GENERATION COMPLETED:")
    print(f"• Raw Audio:     {results['raw_audio_path']}")
    print(f"• Refined Audio: {results['refined_audio_path']}")
    print("="*50)


if __name__ == "__main__":
    main()

