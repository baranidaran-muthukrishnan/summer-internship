# Speech-to-Text Transcription

A simple command-line tool for converting a recording or a microphone phrase into text. It uses the `SpeechRecognition` package and Google's online speech recognition service, so an internet connection is required when transcribing.

## Setup

```powershell
cd speech-to-text-transcription
python -m pip install -r requirements.txt
```

## Transcribe a recording

`SpeechRecognition` supports uncompressed WAV, AIFF, and FLAC files directly.

```powershell
python transcribe.py --audio recording.wav --output transcript.txt
```

Choose another language with a BCP-47 code:

```powershell
python transcribe.py --audio recording.wav --language en-IN
```

## Transcribe from the microphone

```powershell
python transcribe.py --microphone --timeout 8 --phrase-limit 30
```

The microphone option needs PyAudio and microphone permission. If the audio is an MP3 or M4A file, convert it to WAV, AIFF, or FLAC first.

## Privacy and reliability

The default recognizer sends audio to Google's recognition service. Avoid sensitive recordings unless that is appropriate for your use case. Accuracy depends on microphone quality, background noise, accent, language setting, and speech clarity.
