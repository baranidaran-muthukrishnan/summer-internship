#!/usr/bin/env python3
"""Transcribe a WAV, AIFF, or FLAC recording with SpeechRecognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def get_recognizer():
    try:
        import speech_recognition as sr
    except ImportError as error:
        raise RuntimeError("SpeechRecognition is not installed. Run: pip install -r requirements.txt") from error
    return sr, sr.Recognizer()


def transcribe_file(audio_path: Path, language: str) -> str:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    sr, recognizer = get_recognizer()
    try:
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
    except ValueError as error:
        raise ValueError("Use an uncompressed WAV, AIFF, or FLAC recording. Convert MP3/M4A first.") from error
    return recognize(recognizer, audio, language, sr)


def transcribe_microphone(language: str, timeout: float | None, phrase_limit: float | None) -> str:
    sr, recognizer = get_recognizer()
    try:
        with sr.Microphone() as source:
            print("Calibrating microphone for ambient noise…", file=sys.stderr)
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening…", file=sys.stderr)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
    except OSError as error:
        raise RuntimeError("Could not access a microphone. Check permissions and install PyAudio.") from error
    return recognize(recognizer, audio, language, sr)


def recognize(recognizer, audio, language: str, sr) -> str:
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError as error:
        raise RuntimeError("Speech was not understood. Try a clearer recording or reduce background noise.") from error
    except sr.RequestError as error:
        raise RuntimeError(f"The Google recognition service could not be reached: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a recording or microphone input to text.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Path to a WAV, AIFF, or FLAC recording")
    source.add_argument("--microphone", action="store_true", help="Record one phrase using the default microphone")
    parser.add_argument("--language", default="en-US", help="BCP-47 language code (default: en-US)")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds to wait for speech when using --microphone")
    parser.add_argument("--phrase-limit", type=float, default=None, help="Maximum recorded seconds when using --microphone")
    parser.add_argument("--output", type=Path, help="Optional text file for the transcript")
    args = parser.parse_args()

    try:
        text = transcribe_microphone(args.language, args.timeout, args.phrase_limit) if args.microphone else transcribe_file(args.audio, args.language)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved transcript to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
