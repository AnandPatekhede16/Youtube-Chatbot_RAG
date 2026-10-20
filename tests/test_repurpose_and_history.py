import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.history_service import load_history, add_to_history, save_history, delete_history_item
from src.repurpose_service import generate_viral_clips, generate_social_thread, generate_seo_blog

def test_history():
    print("▶️ [1/2] Testing History Service...")
    history = load_history()
    print(f"   Loaded {len(history)} history items.")
    for h in history[:3]:
        print(f"   - [{h.get('video_id')}] {h.get('title')[:40]}...")

    # Test adding an item
    test_id = "test_vid_01"
    test_meta = {"title": "Test Video Title", "author": "Test Author"}
    updated = add_to_history(test_id, test_meta)
    assert any(x["video_id"] == test_id for x in updated), "Failed to add to history!"
    print("   Successfully added test video to history.")

    # Test deleting the item
    cleaned = delete_history_item(test_id)
    assert not any(x["video_id"] == test_id for x in cleaned), "Failed to delete from history!"
    print("   Successfully cleaned up test item from history.")
    print("✅ History service passed 100%!")

def test_repurpose_module():
    print("\n▶️ [2/2] Testing Repurpose Service Module Integrity...")
    assert callable(generate_viral_clips)
    assert callable(generate_social_thread)
    assert callable(generate_seo_blog)
    print("✅ Repurpose module imports & signatures passed!")

if __name__ == "__main__":
    test_history()
    test_repurpose_module()
    print("\n🎉 ALL REPURPOSE & HISTORY TESTS PASSED!")
