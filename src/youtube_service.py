import re
import json
import logging
from typing import Dict, Any, List, Optional
import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from src.config import TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract YouTube 11-character video ID from varied URL formats or raw ID.
    Supports:
    - Standard: https://www.youtube.com/watch?v=VIDEO_ID
    - Shortened: https://youtu.be/VIDEO_ID
    - Shorts: https://www.youtube.com/shorts/VIDEO_ID
    - Embed: https://www.youtube.com/embed/VIDEO_ID
    - Mobile: https://m.youtube.com/watch?v=VIDEO_ID
    - Raw ID: VIDEO_ID
    """
    text = url_or_id.strip()
    if re.fullmatch(r"^[a-zA-Z0-9_-]{11}$", text):
        return text

    patterns = [
        r"(?:v=|\/)([a-zA-Z0-9_-]{11})(?:[&?]|$)",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
            
    return None

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def fetch_video_metadata(video_id: str) -> Dict[str, Any]:
    """
    Fetch public metadata (title, author, thumbnail) via YouTube's oEmbed endpoint.
    No API key required!
    """
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    
    fallback_metadata = {
        "video_id": video_id,
        "title": f"YouTube Video ({video_id})",
        "author": "YouTube Creator",
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
    }

    try:
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "video_id": video_id,
                "title": data.get("title", fallback_metadata["title"]),
                "author": data.get("author_name", fallback_metadata["author"]),
                "thumbnail_url": data.get("thumbnail_url", fallback_metadata["thumbnail_url"]),
                "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            }
    except Exception as e:
        logger.warning(f"Could not fetch oEmbed metadata for {video_id}: {e}")

    return fallback_metadata

def search_nested_dict(partial, search_key):
    """Recursively search for key in dictionaries/lists."""
    if isinstance(partial, dict):
        for k, v in partial.items():
            if k == search_key:
                yield v
            else:
                yield from search_nested_dict(v, search_key)
    elif isinstance(partial, list):
        for item in partial:
            yield from search_nested_dict(item, search_key)

def extract_fallback_chapters_and_description(video_id: str) -> List[Dict[str, Any]]:
    """
    Recovers video chapter markers, timestamps, and description when 
    subtitles/captions are disabled by the video owner.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text

        pos2 = html.find("var ytInitialData = ")
        if pos2 == -1:
            pos2 = html.find("ytInitialData = ")
        if pos2 == -1:
            return []

        start2 = pos2 + html[pos2:].find("{")
        end2 = html.find(";</script>", start2)
        if end2 == -1:
            end2 = html.find(";\n", start2)
        data = json.loads(html[start2:end2])

        # 1. Look for macroMarkersListItemRenderer (chapters)
        chapters = list(search_nested_dict(data, "macroMarkersListItemRenderer"))
        snippets = []
        for ch in chapters:
            title = ch.get("title", {}).get("simpleText", "").strip()
            time_str = ch.get("timeDescription", {}).get("simpleText", "").strip()
            if not title:
                continue
            parts = [int(p) for p in time_str.split(":")]
            if len(parts) == 3:
                sec = parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2:
                sec = parts[0]*60 + parts[1]
            else:
                sec = 0
            snippets.append({
                "text": f"Video Chapter at [{time_str}]: {title}",
                "start": float(sec),
                "duration": 60.0,
                "is_chapter": True
            })

        if snippets:
            logger.info(f"Recovered {len(snippets)} chapter timestamps as fallback for {video_id}.")
            return snippets

        # 2. Look for description with timestamps
        desc_list = list(search_nested_dict(data, "attributedDescriptionBodyText"))
        if desc_list:
            desc_text = desc_list[0].get("content", "").strip()
            if desc_text:
                ts_re = re.compile(r"(?:([0-9]{1,2}):)?([0-5]?[0-9]):([0-5][0-9])")
                for line in desc_text.split("\n"):
                    line = line.strip()
                    m = ts_re.search(line)
                    if m and len(line) > 5:
                        ts_str = m.group(0)
                        parts = [int(p) for p in ts_str.split(":")]
                        sec = parts[0]*3600 + parts[1]*60 + parts[2] if len(parts)==3 else parts[0]*60 + parts[1]
                        snippets.append({
                            "text": f"Chapter: {line}",
                            "start": float(sec),
                            "duration": 60.0,
                            "is_chapter": True
                        })
                if snippets:
                    return snippets

                # If no timestamps, return structured description
                return [{
                    "text": f"Video Description & Overview: {desc_text[:4000]}",
                    "start": 0.0,
                    "duration": 120.0,
                    "is_chapter": True
                }]

    except Exception as e:
        logger.warning(f"Error during fallback chapter extraction for {video_id}: {e}")

    return []

def get_transcript_snippets(
    video_id: str, 
    groq_api_key: Optional[str] = None, 
    openai_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve raw transcript chunks with start timestamps and duration.
    Caches results locally to avoid redundant network calls.
    
    Cascading Multi-Tier Fallback:
    1. Native YouTube Captions (YouTubeTranscriptApi - multi-lingual)
    2. Audio AI Fallback (yt-dlp stream + Groq Whisper / OpenAI Whisper) for 100% video coverage
    3. Structural Video Chapters & Description Markers
    """
    cache_file = TRANSCRIPTS_DIR / f"{video_id}.json"

    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception as e:
            logger.warning(f"Error reading transcript cache: {e}")

    raw_snippets: List[Dict[str, Any]] = []

    # 1. Try modern youtube-transcript-api (v1.x)
    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB", "hi", "hi-Latn"])
            for s in fetched.snippets:
                raw_snippets.append({
                    "text": s.text if hasattr(s, "text") else s.get("text", ""),
                    "start": float(s.start if hasattr(s, "start") else s.get("start", 0.0)),
                    "duration": float(s.duration if hasattr(s, "duration") else s.get("duration", 0.0))
                })
        except Exception:
            # Try listing all available languages (e.g. Hindi, Spanish, auto-translated)
            t_list = api.list(video_id)
            available = list(t_list)
            if available:
                fetched = available[0].fetch()
                for s in fetched.snippets:
                    raw_snippets.append({
                        "text": s.text if hasattr(s, "text") else s.get("text", ""),
                        "start": float(s.start if hasattr(s, "start") else s.get("start", 0.0)),
                        "duration": float(s.duration if hasattr(s, "duration") else s.get("duration", 0.0))
                    })
    except Exception as e:
        logger.debug(f"Captions fetch attempt failed: {e}")

    # 2. Try legacy get_transcript (v0.x)
    if not raw_snippets:
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB", "hi"])
                for s in data:
                    raw_snippets.append({
                        "text": s["text"],
                        "start": float(s["start"]),
                        "duration": float(s.get("duration", 0.0))
                    })
        except Exception as e:
            logger.debug(f"Legacy fetch attempt failed: {e}")

    # 3. Topic 1: 100% Video Coverage - Audio AI Transcription Fallback (Whisper)
    if not raw_snippets:
        try:
            from src.audio_transcriber import transcribe_audio_fallback
            logger.info(f"Subtitles disabled for '{video_id}'. Activating Whisper Audio Fallback...")
            audio_snippets = transcribe_audio_fallback(
                video_id=video_id, 
                groq_api_key=groq_api_key, 
                openai_api_key=openai_api_key
            )
            if audio_snippets:
                raw_snippets = audio_snippets
                logger.info(f"Successfully recovered transcript via Whisper Audio Fallback ({len(raw_snippets)} segments)!")
        except Exception as e:
            logger.warning(f"Audio transcription fallback encountered an issue: {e}")

    # 4. Resilient Fallback: Chapter Markers & Video Description
    if not raw_snippets:
        logger.info(f"Audio fallback unavailable or incomplete for '{video_id}'. Activating chapter & description fallback...")
        raw_snippets = extract_fallback_chapters_and_description(video_id)

    if not raw_snippets:
        raise RuntimeError(
            f"Could not retrieve transcript, audio speech, or chapters for video '{video_id}'. "
            f"The video may be private, age-restricted, or unavailable."
        )

    # Save to local cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_snippets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to cache transcript to disk: {e}")

    return raw_snippets
