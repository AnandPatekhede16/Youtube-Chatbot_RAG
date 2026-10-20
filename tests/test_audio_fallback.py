import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_transcriber import download_audio_stream, transcribe_audio_fallback
from src.youtube_service import get_transcript_snippets

def test_audio_fallback_import():
    print("Testing audio fallback imports and module integrity...")
    assert callable(download_audio_stream)
    assert callable(transcribe_audio_fallback)
    assert callable(get_transcript_snippets)
    print("✅ Audio fallback module & service integrity verified!")

if __name__ == "__main__":
    test_audio_fallback_import()
