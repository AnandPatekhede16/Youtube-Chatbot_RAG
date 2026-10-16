import re
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from src.config import COMMENTS_DIR

logger = logging.getLogger(__name__)

TIMESTAMP_PATTERN = re.compile(r"\b(?:([0-9]{1,2}):)?([0-5]?[0-9]):([0-5][0-9])\b")

def parse_timestamps_from_text(text: str, video_id: str) -> List[Dict[str, Any]]:
    """
    Extracts timestamps like 02:45 or 1:15:30 mentioned in a comment,
    converts to total seconds and generates clickable YouTube deep-link URLs.
    """
    results = []
    for match in TIMESTAMP_PATTERN.finditer(text):
        ts_str = match.group(0)
        parts = [int(p) for p in ts_str.split(":")]
        if len(parts) == 3:
            total_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            total_sec = parts[0] * 60 + parts[1]
        
        results.append({
            "timestamp": ts_str,
            "seconds": total_sec,
            "url": f"https://youtu.be/{video_id}?t={total_sec}"
        })
    return results

def search_dict(partial, search_key):
    """Recursively search for a key in nested dictionaries and lists."""
    if isinstance(partial, dict):
        for k, v in partial.items():
            if k == search_key:
                yield v
            else:
                yield from search_dict(v, search_key)
    elif isinstance(partial, list):
        for item in partial:
            yield from search_dict(item, search_key)

def fetch_video_comments(video_id: str, max_comments: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches real public YouTube comments via InnerTube without requiring API keys.
    Caches results to disk in storage/comments/{video_id}.json.
    """
    cache_file = COMMENTS_DIR / f"{video_id}.json"

    # 1. Check local cache
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached:
                    return cached
        except Exception as e:
            logger.warning(f"Failed to read cached comments for {video_id}: {e}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()

    comments: List[Dict[str, Any]] = []

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        resp = session.get(url, headers=headers, timeout=10)
        html = resp.text

        # Extract ytcfg
        pos = html.find("ytcfg.set({")
        if pos == -1:
            logger.warning("Could not find ytcfg in page HTML")
            return []
        start = pos + len("ytcfg.set(")
        end = html.find(");", start)
        ytcfg = json.loads(html[start:end])

        # Extract ytInitialData
        pos2 = html.find("var ytInitialData = ")
        if pos2 == -1:
            pos2 = html.find("ytInitialData = ")
        if pos2 == -1:
            logger.warning("Could not find ytInitialData in page HTML")
            return []

        start2 = pos2 + html[pos2:].find("{")
        end2 = html.find(";</script>", start2)
        if end2 == -1:
            end2 = html.find(";\n", start2)
        init_data = json.loads(html[start2:end2])

        # Find comment section continuation token
        comment_token = None
        for item in search_dict(init_data, "itemSectionRenderer"):
            if item.get("sectionIdentifier") == "comment-item-section":
                tokens = list(search_dict(item, "continuationCommand"))
                if tokens:
                    comment_token = tokens[0].get("token")
                    break

        if not comment_token:
            logger.info(f"No comment continuation token found for {video_id} (comments might be disabled).")
            return []

        # Request comments via InnerTube /next endpoint
        api_url = "https://www.youtube.com/youtubei/v1/next"
        payload = {
            "context": ytcfg.get("INNERTUBE_CONTEXT", {}),
            "continuation": comment_token
        }
        api_resp = session.post(api_url, json=payload, headers=headers, timeout=10)
        next_data = api_resp.json()

        # Parse from modern frameworkUpdates mutations
        mutations = next_data.get("frameworkUpdates", {}).get("entityBatchUpdate", {}).get("mutations", [])
        for m in mutations:
            payload = m.get("payload", {})
            if "commentEntityPayload" in payload:
                c = payload["commentEntityPayload"]
                author = c.get("author", {}).get("displayName", "Anonymous")
                avatar = c.get("author", {}).get("avatarThumbnailUrl", "")
                text = c.get("properties", {}).get("content", {}).get("content", "").strip()
                published = c.get("properties", {}).get("publishedTime", "")
                toolbar = c.get("toolbar", {})
                likes = toolbar.get("likeCountNotliked", "0")

                if text:
                    ts_links = parse_timestamps_from_text(text, video_id)
                    comments.append({
                        "author": author,
                        "avatar": avatar,
                        "text": text,
                        "published": published,
                        "likes": likes,
                        "timestamps": ts_links
                    })

                if len(comments) >= max_comments:
                    break

        # Fallback to legacy commentRenderer if mutations was empty
        if not comments:
            for c in search_dict(next_data, "commentRenderer"):
                author = c.get("authorText", {}).get("simpleText", "Anonymous")
                text = "".join(run.get("text", "") for run in c.get("contentText", {}).get("runs", []))
                votes = c.get("voteCount", {}).get("simpleText", "0")
                time_text = "".join(run.get("text", "") for run in c.get("publishedTimeText", {}).get("runs", []))
                if text:
                    ts_links = parse_timestamps_from_text(text, video_id)
                    comments.append({
                        "author": author,
                        "avatar": "",
                        "text": text,
                        "published": time_text,
                        "likes": votes,
                        "timestamps": ts_links
                    })
                if len(comments) >= max_comments:
                    break

    except Exception as e:
        logger.error(f"Error fetching comments for video {video_id}: {e}")

    # Cache to disk
    if comments:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache comments to disk: {e}")

    return comments
