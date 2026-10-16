import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.youtube_service import extract_video_id, fetch_video_metadata, get_transcript_snippets
from src.text_processor import TimestampAwareChunker
from src.vectorstore import build_or_load_vector_store, is_video_indexed
from src.rag_engine import YouTubeRAG
from src.config import validate_environment, AI_PROVIDER

def test_pipeline():
    print(f"▶️ [1/5] Testing Video ID extraction...")
    test_urls = [
        ("https://www.youtube.com/watch?v=Gfr50f6ZBvo", "Gfr50f6ZBvo"),
        ("https://youtu.be/Gfr50f6ZBvo", "Gfr50f6ZBvo"),
        ("https://www.youtube.com/shorts/Gfr50f6ZBvo", "Gfr50f6ZBvo"),
        ("Gfr50f6ZBvo", "Gfr50f6ZBvo")
    ]
    for url, expected in test_urls:
        vid = extract_video_id(url)
        assert vid == expected, f"Expected {expected}, got {vid}"
    print("✅ Video ID extraction passed.")

    video_id = "Gfr50f6ZBvo"

    print("\n▶️ [2/5] Testing Metadata fetching...")
    meta = fetch_video_metadata(video_id)
    print(f"   Title: {meta.get('title')}")
    print(f"   Author: {meta.get('author')}")
    assert "video_id" in meta
    print("✅ Metadata fetching passed.")

    print("\n▶️ [3/5] Testing Transcript retrieval...")
    snippets = get_transcript_snippets(video_id)
    assert len(snippets) > 0, "No snippets retrieved!"
    print(f"   Retrieved {len(snippets)} snippets.")
    print("✅ Transcript retrieval passed.")

    print("\n▶️ [4/5] Testing Timestamp-Aware Chunking...")
    chunker = TimestampAwareChunker(target_chunk_size=1000, overlap_snippets=2)
    docs = chunker.create_documents(snippets, video_id)
    assert len(docs) > 0
    first_doc = docs[0]
    print(f"   Created {len(docs)} document chunks.")
    print(f"   First chunk metadata: {first_doc.metadata}")
    assert "timestamp" in first_doc.metadata
    assert "timestamp_url" in first_doc.metadata
    print("✅ Chunking passed.")

    print(f"\n▶️ [5/5] Testing Vector Store and RAG Q&A with {AI_PROVIDER.upper()}...")
    if not validate_environment():
        print(f"⚠️ {AI_PROVIDER.upper()}_API_KEY not configured, skipping live API call.")
        return

    vector_store = build_or_load_vector_store(video_id, docs, provider=AI_PROVIDER)
    assert is_video_indexed(video_id, provider=AI_PROVIDER)
    print(f"   Vector store successfully built & cached on disk for {AI_PROVIDER}!")

    rag = YouTubeRAG(vector_store, meta, provider=AI_PROVIDER)
    question = "Who is Demis and what did he discuss?"
    print(f"   Asking question: '{question}'")
    result = rag.ask(question)
    
    print("\n--- AI Response ---")
    print(result["answer"])
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"[{s['timestamp']}] ({s['url']}) : {s['text'][:80]}...")
    
    assert len(result["answer"]) > 20
    print("\n🎉 ALL 5 PIPELINE TESTS PASSED 100% WITH LIVE AI!")

if __name__ == "__main__":
    test_pipeline()
