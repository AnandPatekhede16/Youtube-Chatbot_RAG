import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import (
    TEMP_AUDIO_DIR,
    GROQ_API_KEY,
    OPENAI_API_KEY
)

logger = logging.getLogger(__name__)

def download_audio_stream(video_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Downloads lightweight audio stream for a YouTube video using yt-dlp.
    Selects lower bitrate (mono/low bandwidth) to ensure fast download and minimal size.
    """
    import yt_dlp
    
    out_dir = output_dir or TEMP_AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'format': 'worstaudio[ext=m4a]/worstaudio[ext=mp3]/worstaudio/bestaudio[ext=m4a]/bestaudio',
        'outtmpl': out_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        # Max file size 25MB for API tier safety
        'max_filesize': 25 * 1024 * 1024,
    }
    
    try:
        logger.info(f"Downloading audio stream for video '{video_id}'...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            if filename and Path(filename).exists():
                logger.info(f"Audio downloaded successfully: {filename} ({os.path.getsize(filename)} bytes)")
                return Path(filename)
    except Exception as e:
        logger.warning(f"yt-dlp audio download failed for {video_id}: {e}")
        # Try best audio if worstaudio failed
        try:
            ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
                if filename and Path(filename).exists():
                    return Path(filename)
        except Exception as e2:
            logger.warning(f"Retry audio download also failed: {e2}")
            
    return None

def transcribe_with_groq(audio_path: Path, api_key: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribes audio using Groq Whisper API (whisper-large-v3-turbo).
    Returns timestamped snippet segments.
    """
    active_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "").strip()
    if not active_key:
        logger.debug("No GROQ_API_KEY provided; skipping Groq Whisper.")
        return None
        
    try:
        from groq import Groq
        client = Groq(api_key=active_key)
        
        logger.info(f"Transcribing audio with Groq Whisper API: {audio_path.name}...")
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_path.name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
        snippets = []
        segments = getattr(transcription, "segments", None)
        if segments is None and isinstance(transcription, dict):
            segments = transcription.get("segments", [])
            
        if segments:
            for s in segments:
                text = (s.get("text") if isinstance(s, dict) else getattr(s, "text", "")).strip()
                start = float(s.get("start") if isinstance(s, dict) else getattr(s, "start", 0.0))
                end = float(s.get("end") if isinstance(s, dict) else getattr(s, "end", start + 5.0))
                if text:
                    snippets.append({
                        "text": text,
                        "start": round(start, 2),
                        "duration": round(max(1.0, end - start), 2),
                        "is_fallback": True,
                        "source": "groq_whisper"
                    })
                    
        if snippets:
            logger.info(f"Groq Whisper transcribed {len(snippets)} speech segments successfully!")
            return snippets
            
    except Exception as e:
        logger.warning(f"Groq Whisper transcription failed: {e}")
        
    return None

def transcribe_with_openai(audio_path: Path, api_key: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribes audio using OpenAI Whisper API (whisper-1).
    """
    active_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "").strip()
    if not active_key:
        logger.debug("No OPENAI_API_KEY provided; skipping OpenAI Whisper.")
        return None
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=active_key)
        
        logger.info(f"Transcribing audio with OpenAI Whisper API: {audio_path.name}...")
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
        snippets = []
        segments = getattr(transcription, "segments", None)
        if segments is None and isinstance(transcription, dict):
            segments = transcription.get("segments", [])
            
        if segments:
            for s in segments:
                text = (s.get("text") if isinstance(s, dict) else getattr(s, "text", "")).strip()
                start = float(s.get("start") if isinstance(s, dict) else getattr(s, "start", 0.0))
                end = float(s.get("end") if isinstance(s, dict) else getattr(s, "end", start + 5.0))
                if text:
                    snippets.append({
                        "text": text,
                        "start": round(start, 2),
                        "duration": round(max(1.0, end - start), 2),
                        "is_fallback": True,
                        "source": "openai_whisper"
                    })
                    
        if snippets:
            logger.info(f"OpenAI Whisper transcribed {len(snippets)} speech segments successfully!")
            return snippets
            
    except Exception as e:
        logger.warning(f"OpenAI Whisper transcription failed: {e}")
        
    return None

def transcribe_audio_fallback(
    video_id: str, 
    groq_api_key: Optional[str] = None, 
    openai_api_key: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    High-level orchestration for Topic 1 (100% Video Coverage via Audio Fallback):
    1. Downloads audio via yt-dlp.
    2. Runs Groq Whisper (ultra-fast, free tier).
    3. Falls back to OpenAI Whisper if configured.
    4. Automatically cleans up temp audio file to preserve disk space.
    """
    audio_path = None
    try:
        # Check if any transcriber key is available first
        has_groq = bool(groq_api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "").strip())
        has_openai = bool(openai_api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "").strip())
        
        if not (has_groq or has_openai):
            logger.info("No Whisper API keys (Groq or OpenAI) available for audio transcription fallback.")
            return None
            
        audio_path = download_audio_stream(video_id)
        if not audio_path or not audio_path.exists():
            return None
            
        # 1. Try Groq Whisper (Top priority: fast & free)
        if has_groq:
            snippets = transcribe_with_groq(audio_path, api_key=groq_api_key)
            if snippets:
                return snippets
                
        # 2. Try OpenAI Whisper
        if has_openai:
            snippets = transcribe_with_openai(audio_path, api_key=openai_api_key)
            if snippets:
                return snippets
                
    except Exception as e:
        logger.warning(f"Error in audio transcription fallback pipeline: {e}")
    finally:
        # Always remove temporary audio file
        if audio_path and audio_path.exists():
            try:
                os.remove(audio_path)
                logger.debug(f"Cleaned up temporary audio: {audio_path}")
            except Exception as e:
                logger.warning(f"Could not remove temp audio file {audio_path}: {e}")
                
    return None
