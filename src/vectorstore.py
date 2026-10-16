import logging
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import (
    VECTORS_DIR,
    AI_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    DEFAULT_TOP_K
)

logger = logging.getLogger(__name__)

def get_embeddings(provider: str = AI_PROVIDER, api_key: Optional[str] = None):
    """Initializes and returns embeddings instance based on provider."""
    provider = provider.lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        active_key = api_key or GEMINI_API_KEY
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=active_key
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        active_key = api_key or OPENAI_API_KEY
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=active_key
        )

def get_vector_store_path(video_id: str, provider: str = AI_PROVIDER) -> Path:
    """Returns directory path where the FAISS index is saved for this video and provider."""
    return VECTORS_DIR / f"{video_id}_{provider}"

def is_video_indexed(video_id: str, provider: str = AI_PROVIDER) -> bool:
    """Check if a vector index already exists on disk for this video ID and provider."""
    index_dir = get_vector_store_path(video_id, provider)
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()

def build_or_load_vector_store(
    video_id: str, 
    documents: Optional[List[Document]] = None,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> FAISS:
    """
    Loads existing FAISS vector store from disk cache if present.
    Otherwise builds a new one from documents and caches it locally.
    """
    index_dir = get_vector_store_path(video_id, provider)
    embeddings = get_embeddings(provider=provider, api_key=api_key)

    if is_video_indexed(video_id, provider):
        logger.info(f"Loading cached FAISS index for video: {video_id} ({provider})")
        vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store

    if not documents:
        raise ValueError(
            f"No index found for video '{video_id}' ({provider}) and no documents were provided to build one."
        )

    logger.info(f"Building new FAISS vector store with {len(documents)} chunks for video: {video_id} ({provider})...")
    
    # Safe batching to respect free tier rate limits (e.g. Gemini 100 req/min)
    batch_size = 50
    vector_store = None

    import time
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        for attempt in range(5):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    logger.warning(f"Rate limit hit during embedding batch. Waiting 15s (Attempt {attempt+1}/5)...")
                    time.sleep(15)
                else:
                    raise e

    # Save to disk
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    logger.info(f"Saved FAISS index to {index_dir}")

    return vector_store

def get_retriever(vector_store: FAISS, top_k: int = DEFAULT_TOP_K):
    """
    Returns a configured retriever from the FAISS vector store.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
