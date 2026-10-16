# Standard unit tests without external pytest dependency
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voice_player_service import (
    strip_markdown_for_speech,
    render_sync_player_html,
    render_voice_input_html,
    render_tts_player_html
)
from src.study_service import (
    generate_study_notes_markdown,
    _clean_json_text
)

def test_strip_markdown_for_speech():
    raw = "**Hello!** Check out [14:20](https://youtu.be/123?t=860) and `code` here: http://example.com"
    clean = strip_markdown_for_speech(raw)
    assert "https://" not in clean
    assert "http://" not in clean
    assert "*" not in clean
    assert "`" not in clean
    assert "14:20" in clean
    assert "Hello!" in clean

def test_render_sync_player_html():
    html = render_sync_player_html("4Vz6L8B73i4", start_seconds=120)
    assert "4Vz6L8B73i4" in html
    assert "start=120" in html
    assert "<iframe" in html

def test_render_voice_input_html():
    html = render_voice_input_html()
    assert "Ask with Voice" in html
    assert "SpeechRecognition" in html
    assert "copyTranscript" in html

def test_render_tts_player_html():
    html = render_tts_player_html("This is a test answer for TTS.", button_id="test1")
    assert "tts-btn-test1" in html
    assert "SpeechSynthesisUtterance" in html
    assert "window.speechSynthesis" in html

def test_clean_json_text():
    raw_fenced = "```json\n[{\"id\": 1}]\n```"
    cleaned = _clean_json_text(raw_fenced)
    assert cleaned == '[{"id": 1}]'

def test_generate_study_notes_markdown():
    meta = {"title": "Test Video Title", "author": "Test Author", "video_id": "test_vid"}
    summary = "This is a summary of the video."
    quiz = [{"id": 1, "question": "What is 2+2?", "options": ["3", "4", "5"], "answer_index": 1, "explanation": "Math", "timestamp": "01:00"}]
    flashcards = [{"id": 1, "front": "Concept 1", "back": "Explanation 1", "tag": "Focus", "timestamp": "02:00"}]
    mindmap = "flowchart TD\nA --> B"
    comments = "Audience loved the clarity."

    md = generate_study_notes_markdown(meta, summary, quiz, flashcards, mindmap, comments)
    assert "Test Video Title" in md
    assert "Test Author" in md
    assert "Executive Summary" in md
    assert "Concept Flashcards" in md
    assert "Comprehension Quiz" in md
    assert "Concept Mind Map" in md
    assert "flowchart TD" in md

if __name__ == "__main__":
    test_strip_markdown_for_speech()
    test_render_sync_player_html()
    test_render_voice_input_html()
    test_render_tts_player_html()
    test_clean_json_text()
    test_generate_study_notes_markdown()
    print("All study and voice tests passed successfully! [OK]")
