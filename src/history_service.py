import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import HISTORY_FILE, TRANSCRIPTS_DIR
from src.youtube_service import fetch_video_metadata

logger = logging.getLogger(__name__)

def load_history() -> List[Dict[str, Any]]:
    """
    Loads analysis history from storage/history.json.
    If the file does not exist, automatically backfills from existing cached transcripts in storage/.
    """
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning(f"Could not read history.json: {e}")

    # Backfill from cached transcripts if history.json is absent or empty
    backfilled = []
    if TRANSCRIPTS_DIR.exists():
        try:
            for json_file in TRANSCRIPTS_DIR.glob("*.json"):
                vid = json_file.stem
                if len(vid) == 11:
                    mtime = datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
                    meta = fetch_video_metadata(vid)
                    backfilled.append({
                        "video_id": vid,
                        "title": meta.get("title", f"Video ({vid})"),
                        "author": meta.get("author", "YouTube Creator"),
                        "thumbnail_url": meta.get("thumbnail_url", f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"),
                        "last_analyzed": mtime
                    })
            # Sort by last_analyzed descending
            backfilled.sort(key=lambda x: x.get("last_analyzed", ""), reverse=True)
            if backfilled:
                save_history(backfilled)
        except Exception as e:
            logger.warning(f"Error backfilling history: {e}")

    return backfilled

def save_history(history_items: List[Dict[str, Any]]) -> None:
    """Saves the history list to storage/history.json."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")

def add_to_history(video_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Adds or updates a video in the history list and moves it to the top.
    Keeps up to the most recent 50 entries.
    """
    items = load_history()
    now_iso = datetime.now().isoformat()
    meta = metadata or fetch_video_metadata(video_id)

    # Filter out existing entry for this video_id
    filtered = [item for item in items if item.get("video_id") != video_id]

    new_entry = {
        "video_id": video_id,
        "title": meta.get("title", f"Video ({video_id})"),
        "author": meta.get("author", "YouTube Creator"),
        "thumbnail_url": meta.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
        "last_analyzed": now_iso
    }

    # Prepend new entry
    updated = [new_entry] + filtered
    # Limit to 50 items
    updated = updated[:50]
    save_history(updated)
    return updated

def delete_history_item(video_id: str) -> List[Dict[str, Any]]:
    """Removes a video entry from history."""
    items = load_history()
    updated = [item for item in items if item.get("video_id") != video_id]
    save_history(updated)
    return updated

def clear_all_history() -> None:
    """Clears the history list."""
    save_history([])
