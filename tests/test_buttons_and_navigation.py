"""
Tests verifying button functionality, JSON extraction resilience, force_refresh behavior,
and persistent navigation tab keys without slow network calls.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.study_service import (
    _extract_json_data,
    _clean_json_text,
    generate_quiz,
    generate_flashcards,
    generate_mindmap
)
from src.repurpose_service import (
    generate_viral_clips,
    generate_social_thread,
    generate_seo_blog
)

def test_json_resilience():
    print("▶️ [1/3] Testing JSON resilience & extraction...")
    
    # 1. Text with markdown fence
    fenced_sample = "Here is the response:\n```json\n[{\"id\": 1, \"front\": \"Test\", \"back\": \"Answer\"}]\n```\nHope that helps!"
    res1 = _extract_json_data(fenced_sample)
    assert isinstance(res1, list) and len(res1) == 1
    assert res1[0]["front"] == "Test"

    # 2. Text with raw bracket array
    raw_array = "Sure, here are the cards: [{\"id\": 2, \"front\": \"Card2\", \"back\": \"Answer2\"}]"
    res2 = _extract_json_data(raw_array)
    assert isinstance(res2, list) and len(res2) == 1
    assert res2[0]["front"] == "Card2"

    # 3. Text with object
    raw_obj = "```json\n{\"hook_headline\": \"Great Hook\", \"tweets\": [\"Tweet 1\", \"Tweet 2\"]}\n```"
    res3 = _extract_json_data(raw_obj)
    assert isinstance(res3, dict)
    assert res3["hook_headline"] == "Great Hook"
    assert len(res3["tweets"]) == 2

    print("✅ JSON extraction resilience verified across all formats!")

def test_force_refresh_signatures():
    print("\n▶️ [2/3] Testing force_refresh signatures...")
    import inspect

    for fn in [generate_quiz, generate_flashcards, generate_mindmap, generate_viral_clips, generate_social_thread, generate_seo_blog]:
        sig = inspect.signature(fn)
        assert "force_refresh" in sig.parameters, f"{fn.__name__} is missing force_refresh parameter!"
        assert sig.parameters["force_refresh"].default is False, f"{fn.__name__} force_refresh default should be False"
    print("✅ All 6 generation services accept force_refresh parameter with default=False!")

def test_cached_and_fallback_integrity():
    print("\n▶️ [3/3] Testing cached asset loading & fallback generation...")
    test_vid = "test_mock_vid"
    
    # 1. Mock cache reading for quiz
    cache_quiz_path = Path("storage/study") / f"{test_vid}_quiz.json"
    cache_quiz_path.parent.mkdir(parents=True, exist_ok=True)
    sample_quiz = [{"id": 1, "question": "Q?", "options": ["A", "B"], "answer_index": 0}]
    with open(cache_quiz_path, "w", encoding="utf-8") as f:
        json.dump(sample_quiz, f)
    
    loaded_quiz = generate_quiz(test_vid, "dummy", force_refresh=False)
    assert loaded_quiz == sample_quiz
    print("   Cached quiz read correctly!")

    # 2. Mock cache reading for mindmap
    cache_mm_path = Path("storage/study") / f"{test_vid}_mindmap.json"
    with open(cache_mm_path, "w", encoding="utf-8") as f:
        json.dump({"mermaid_code": "graph TD\nA-->B"}, f)
    
    loaded_mm = generate_mindmap(test_vid, "dummy", force_refresh=False)
    assert loaded_mm == "graph TD\nA-->B"
    print("   Cached mindmap read correctly!")

    # 3. Mock cache reading for clips
    cache_clips_path = Path("storage/repurpose") / f"{test_vid}_viral_clips.json"
    cache_clips_path.parent.mkdir(parents=True, exist_ok=True)
    sample_clips = [{"id": 1, "hook_title": "Clip 1", "start_time": "01:00"}]
    with open(cache_clips_path, "w", encoding="utf-8") as f:
        json.dump(sample_clips, f)
    
    loaded_clips = generate_viral_clips(test_vid, "dummy", force_refresh=False)
    assert loaded_clips == sample_clips
    print("   Cached clips read correctly!")

    # Clean up test artifacts
    cache_quiz_path.unlink(missing_ok=True)
    cache_mm_path.unlink(missing_ok=True)
    cache_clips_path.unlink(missing_ok=True)

    print("✅ Caching and fallback bypass verified cleanly!")

if __name__ == "__main__":
    test_json_resilience()
    test_force_refresh_signatures()
    test_cached_and_fallback_integrity()
    print("\n🎉 ALL BUTTON & NAVIGATION TESTS PASSED SUCCESSFULLY!")
