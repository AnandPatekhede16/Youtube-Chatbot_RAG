import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
VECTORS_DIR = STORAGE_DIR / "vectors"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
COMMENTS_DIR = STORAGE_DIR / "comments"
COMMENTS_ANALYSIS_DIR = STORAGE_DIR / "comments_analysis"
STUDY_DIR = STORAGE_DIR / "study"
TEMP_AUDIO_DIR = STORAGE_DIR / "temp_audio"
REPURPOSE_DIR = STORAGE_DIR / "repurpose"
HISTORY_FILE = STORAGE_DIR / "history.json"

# Ensure storage directories exist
VECTORS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
COMMENTS_DIR.mkdir(parents=True, exist_ok=True)
COMMENTS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REPURPOSE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Provider Settings ("gemini" or "openai")
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").strip().lower()

# Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Groq Whisper Settings (for 100% video coverage audio fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Text Chunking Settings
DEFAULT_CHUNK_SIZE = 1600  # in characters (~250-300 words)
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Settings
DEFAULT_TOP_K = 4

def validate_environment(provider: str = None) -> bool:
    """Check if mandatory environment variables are present for active provider."""
    active_prov = (provider or AI_PROVIDER).lower()
    if active_prov == "gemini":
        return bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"))
    elif active_prov == "openai":
        return bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"))
    return False
