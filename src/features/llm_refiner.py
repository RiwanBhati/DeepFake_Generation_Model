"""
llm_refiner.py
--------------
Step 2 of the DeepFake Generation Pipeline:
Refines target text using an LLM to optimize it for the IndicF5-Hinglish model.

Key Requirements for IndicF5:
1. Hindi words converted to Devanagari script (e.g. "मेरा", "काम", "है").
2. English words preserved in standard English Latin script (e.g. "project", "deadline", "presentation").
3. Conversational punctuation added (commas for pauses, periods, question marks) to produce natural cadence.
"""

import os
import json
import argparse
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You are an expert Hinglish (Hindi + English) linguistic assistant specializing in preparing text for Text-to-Speech (TTS) voice models (specifically IndicF5).

Your task:
Take a raw target Hinglish sentence (which might be typed in Roman script, poorly punctuated, or casual) and convert it into high-quality code-mixed Hinglish optimized for TTS synthesis.

STRICT FORMATTING RULES:
1. SCRIPT RULE:
   - Write all Hindi words in pure DEVANAGARI script (e.g. "मेरा", "है कि", "करेंगे", "क्या हाल है").
   - Keep all English words in standard ENGLISH LATIN script (e.g. "project", "goal", "presentation", "meeting", "complete").
2. NATURAL PUNCTUATION RULE:
   - Insert commas (,) where natural human breathing pauses occur.
   - Insert question marks (?) or periods (.) appropriately to guide the pitch contour of the voice model.
3. PRESERVE MEANING:
   - Keep the original meaning and tone intact; do not add unnecessary fluff.
4. OUTPUT FORMAT:
   - Output ONLY the final refined Hinglish sentence. No quotes, no markdown, no explanation.

Example Input:
"today im working on a new project mera goal hai ki mai isko properly complete karu aur kuch naya seekhu"

Example Output:
Today I'm working on a new project. मेरा goal है कि मैं इसे properly complete करूँ, और कुछ नया सीखूँ।
"""


def refine_text_with_gemini(raw_text: str, api_key: str) -> str:
    """Refines text using Google Gemini API."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\nInput Text to refine:\n{raw_text}",
        )
        return response.text.strip().strip('"')
    except Exception as e:
        print(f"[WARNING] Gemini client call failed: {e}. Trying legacy google.generativeai...")
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{SYSTEM_PROMPT}\n\nInput Text to refine:\n{raw_text}")
            return response.text.strip().strip('"')
        except Exception as e2:
            print(f"[ERROR] Could not query Gemini: {e2}")
            return None


def refine_text_with_openai(raw_text: str, api_key: str, base_url: str = None) -> str:
    """Refines text using OpenAI or Groq/compatible API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        model_name = "llama-3.3-70b-versatile" if base_url and "groq" in base_url else "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Refine this Hinglish text for IndicF5:\n{raw_text}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"[ERROR] OpenAI/Groq call failed: {e}")
        return None


def refine_target_text(raw_text: str) -> dict:
    """
    Main entry point for text refinement.
    Checks environment variables for API keys.
    Falls back gracefully if no key is currently configured.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    refined_text = None

    if gemini_key:
        print("[INFO] Refining target text using Gemini API...")
        refined_text = refine_text_with_gemini(raw_text, gemini_key)
    elif groq_key:
        print("[INFO] Refining target text using Groq API...")
        refined_text = refine_text_with_openai(raw_text, groq_key, base_url="https://api.groq.com/openai/v1")
    elif openai_key:
        print("[INFO] Refining target text using OpenAI API...")
        refined_text = refine_text_with_openai(raw_text, openai_key)
    else:
        print("[NOTICE] No LLM API key detected in .env (GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY).")
        print("[NOTICE] Using raw input text as baseline for now. You can add your API key in .env anytime.")
        refined_text = raw_text

    result = {
        "raw_text": raw_text,
        "refined_text": refined_text or raw_text
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Refine Hinglish target text with LLM for IndicF5.")
    parser.add_argument("--text", type=str, required=True, help="Raw Hinglish target text to refine")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON result")

    args = parser.parse_args()
    result = refine_target_text(args.text)

    print("\n" + "="*50)
    print("BEFORE LLM (Raw Input):")
    print(result["raw_text"])
    print("-"*50)
    print("AFTER LLM (IndicF5 Refined):")
    print(result["refined_text"])
    print("="*50 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()

