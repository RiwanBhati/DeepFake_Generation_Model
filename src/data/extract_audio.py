"""
extract_audio.py
----------------
Step 1 of the DeepFake Generation Pipeline:
Extracts high-quality reference audio (WAV, 24kHz mono) from an input video file using ffmpeg.
This audio will be used as the voice clone reference for IndicF5-TTS.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Set base project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "raw_videos" / "input_video.mp4"
DEFAULT_AUDIO_OUT = PROJECT_ROOT / "data" / "reference_data" / "ref_audio.wav"


def extract_audio(
    video_path: str,
    output_audio_path: str,
    sample_rate: int = 24000,
    start_time: float = None,
    duration: float = None,
) -> bool:
    """
    Extracts audio track from a video file and converts it to PCM 16-bit mono WAV.
    
    Args:
        video_path: Path to the input video (.mp4, .mov, etc.)
        output_audio_path: Path where the output .wav file should be saved.
        sample_rate: Target sample rate in Hz (default: 24000 Hz for F5-TTS).
        start_time: Optional start time in seconds (e.g. 0.0 to start from beginning).
        duration: Optional duration to extract in seconds (e.g. 10.0 for 10 seconds).
    
    Returns:
        True if extraction succeeded, False otherwise.
    """
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

    # 1. Check if the input video exists
    if not video_path.exists():
        print(f"[ERROR] Input video not found at: {video_path}")
        print(f"[TIP] Place your 10-second video at: {DEFAULT_VIDEO_PATH}")
        return False

    # 2. Ensure destination directory exists
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Build ffmpeg command
    # -vn: disable video recording (extract audio only)
    # -acodec pcm_s16le: standard uncompressed WAV format
    # -ar 24000: 24kHz audio (standard for F5-TTS models)
    # -ac 1: 1 audio channel (mono)
    # -y: overwrite output file if it exists
    cmd = ["ffmpeg", "-y"]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", str(video_path)])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_audio_path)
    ])

    print(f"[INFO] Extracting audio from: {video_path.name}")
    print(f"[INFO] Target format: WAV, {sample_rate}Hz, Mono")

    # Set LD_LIBRARY_PATH so ffmpeg finds liblapack/openblas on cluster
    env = os.environ.copy()
    local_lib = str(Path.home() / ".local" / "lib")
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{local_lib}:{current_ld}" if current_ld else local_lib

    try:
        # Run ffmpeg command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=env
        )
        print(f"[SUCCESS] Reference audio saved to: {output_audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed with error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("[ERROR] ffmpeg is not installed or not in PATH.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract clean reference audio from video.")
    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help="Path to the input video file (default: data/raw_videos/input_video.mp4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_AUDIO_OUT),
        help="Path to save the output WAV file (default: data/reference_data/ref_audio.wav)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (optional)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (e.g. 10.0) (optional)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000 Hz for IndicF5)"
    )

    args = parser.parse_args()
    success = extract_audio(
        video_path=args.video,
        output_audio_path=args.output,
        sample_rate=args.sr,
        start_time=args.start,
        duration=args.duration
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

