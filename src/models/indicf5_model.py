"""
indicf5_model.py
----------------
Model architecture and wrapper for Saravananravi/indicf5-hinglish.
Based on Flow Matching (CFM) with DiT (Diffusion Transformer) backbone.
Synthesizes speech at 24kHz with Vocos vocoder using IndicF5 custom tokenizer.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDICF5_SRC = PROJECT_ROOT / "indicf5_finetune_repo" / "src"
if str(INDICF5_SRC) not in sys.path:
    sys.path.insert(0, str(INDICF5_SRC))

DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "indicf5" / "vocab.txt"

INDICF5_MODEL_CFG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

SAMPLE_RATE = 24000


def get_aligned_reference_text(ref_audio_path: str, user_ref_text: str = None, device: str = "cuda") -> str:
    """
    Uses Whisper Large-v3-Turbo to get an exact phonetic/temporal transcript
    of what was actually spoken in the reference audio.
    F5-TTS requires 100% token-to-spectrogram alignment to avoid static noise or looping.
    """
    try:
        from transformers import pipeline
        import torchaudio
        import torchaudio.transforms as T

        print("[INFO] Aligning reference audio transcript with Whisper Large-v3-Turbo...")
        audio, sr = sf.read(ref_audio_path)
        if sr != 16000:
            t_audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            resampler = T.Resample(sr, 16000)
            audio_16k = resampler(t_audio).squeeze(0).numpy()
        else:
            audio_16k = audio

        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=device)
        result = asr({"sampling_rate": 16000, "raw": audio_16k})
        transcribed = result.get("text", "").strip()
        print(f"[INFO] Spoken reference transcribed as: '{transcribed}'")
        return transcribed if transcribed else user_ref_text
    except Exception as e:
        print(f"[WARN] Automatic reference alignment failed ({e}), falling back to provided ref_text.")
        return user_ref_text


class IndicF5Hinglish(torch.nn.Module):
    """IndicF5 fine-tuned for Hindi-English code-switched (Hinglish) TTS."""

    def __init__(self, vocab_path=None):
        super().__init__()
        vocab_path = vocab_path or str(DEFAULT_VOCAB_PATH)

        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer

        # 1. Custom Indic tokenizer
        self.vocab_char_map, self.vocab_size = get_tokenizer(str(vocab_path), tokenizer="custom")

        # 2. Build DiT backbone
        backbone = DiT(
            **INDICF5_MODEL_CFG,
            text_num_embeds=self.vocab_size,
            mel_dim=100,
        )

        # 3. Build CFM model with custom vocab map
        self.model = CFM(
            transformer=backbone,
            mel_spec_kwargs=dict(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=100,
                target_sample_rate=SAMPLE_RATE,
                mel_spec_type="vocos",
            ),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=self.vocab_char_map,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path="Saravananravi/indicf5-hinglish",
        vocab_path=None,
        device="cuda"
    ):
        """Loads weights from HuggingFace Hub or local cache."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        vocab_path = vocab_path or str(DEFAULT_VOCAB_PATH)
        model = cls(vocab_path=vocab_path)

        if os.path.isdir(pretrained_model_name_or_path):
            ckpt_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(pretrained_model_name_or_path, "model_last.pt")
        else:
            try:
                ckpt_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors",
                )
            except Exception:
                ckpt_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model_last.pt",
                )

        print(f"[INFO] Loading IndicF5 checkpoint: {ckpt_path}")
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            if "ema_model_state_dict" in checkpoint:
                state_dict = checkpoint["ema_model_state_dict"]
                cleaned = {}
                for k, v in state_dict.items():
                    if k.startswith("ema_model."):
                        cleaned[k[len("ema_model."):]] = v
                    elif k in ("initted", "step"):
                        continue
                    else:
                        cleaned[k] = v
                state_dict = cleaned
            else:
                state_dict = checkpoint

        # Remove mel_spec buffers that might clash
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            state_dict.pop(key, None)

        model.model.load_state_dict(state_dict, strict=False)

        dev = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        model = model.to(dev)
        model.eval()
        print(f"[INFO] IndicF5-Hinglish loaded successfully on device: {dev}")
        return model

    def generate(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        nfe_step: int = 32,
        speed: float = 1.0,
        auto_align: bool = True
    ):
        """
        Generate cloned speech audio.
        """
        if ref_audio is None:
            raise ValueError("Reference audio is required for voice cloning!")

        device = next(self.parameters()).device

        # Automatically align reference text to what was actually spoken in the audio
        aligned_ref_text = ref_text
        if auto_align:
            aligned_ref_text = get_aligned_reference_text(ref_audio, user_ref_text=ref_text, device=str(device))

        from f5_tts.infer.utils_infer import (
            infer_process, load_vocoder, preprocess_ref_audio_text
        )

        # Load vocoder (Vocos)
        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

        # Preprocess reference audio and text
        ref_audio_arr, ref_text_proc = preprocess_ref_audio_text(
            ref_audio, aligned_ref_text, show_info=lambda *a: None
        )

        # Synthesize speech using Flow Matching
        audio, sr, _ = infer_process(
            ref_audio_arr,
            ref_text_proc,
            text,
            self.model,
            vocoder,
            mel_spec_type="vocos",
            speed=speed,
            device=str(device),
            nfe_step=nfe_step,
            show_info=lambda *a: None,
        )

        # Normalize audio peak to -1 dB to avoid clipping and keep volume strong
        audio = np.array(audio, dtype=np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_peak = 0.90
            audio = audio * (target_peak / max_val)

        return audio, sr
