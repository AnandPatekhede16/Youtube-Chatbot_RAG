import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
from src.config import validate_environment, AI_PROVIDER
from src.youtube_service import extract_video_id, fetch_video_metadata, get_transcript_snippets
from src.text_processor import TimestampAwareChunker, get_full_transcript_text
from src.vectorstore import build_or_load_vector_store, is_video_indexed
from src.rag_engine import YouTubeRAG

def main():
    parser = argparse.ArgumentParser(description="YouTube AI Companion - Interactive RAG CLI")
    parser.add_argument("--url", type=str, help="YouTube video URL or Video ID", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"🎬 YouTube AI Companion - RAG Assistant (Provider: {AI_PROVIDER.upper()})")
    print("=" * 60)

    if not validate_environment():
        print(f"❌ Error: API Key for {AI_PROVIDER.upper()} is not properly configured in .env file.")
        sys.exit(1)

    url_input = args.url
    if not url_input:
        url_input = input("\n👉 Enter YouTube URL or Video ID: ").strip()

    if not url_input:
        print("❌ No input provided. Exiting.")
        sys.exit(1)

    video_id = extract_video_id(url_input)
    if not video_id:
        print(f"❌ Could not parse valid YouTube Video ID from '{url_input}'")
        sys.exit(1)

    print(f"\n🔍 Processing Video ID: {video_id}")
    
    # 1. Fetch Metadata
    meta = fetch_video_metadata(video_id)
    print(f"📹 Title:  {meta.get('title')}")
    print(f"👤 Author: {meta.get('author')}")

    # 2. Extract Transcript
    try:
        print("📥 Fetching transcript and timestamp cues...")
        snippets = get_transcript_snippets(video_id)
        print(f"✅ Found {len(snippets)} transcript dialogue segments.")
    except Exception as e:
        print(f"❌ Failed to fetch transcript: {e}")
        sys.exit(1)

    full_transcript = get_full_transcript_text(snippets)

    # 3. Vector Storage
    if is_video_indexed(video_id, provider=AI_PROVIDER):
        print(f"⚡ Found existing cached FAISS index on disk for {AI_PROVIDER}. Loading...")
        vector_store = build_or_load_vector_store(video_id, provider=AI_PROVIDER)
    else:
        print("✂️ Chunking text while preserving timestamp metadata...")
        chunker = TimestampAwareChunker(target_chunk_size=1000, overlap_snippets=2)
        docs = chunker.create_documents(snippets, video_id)
        print(f"🧠 Generated {len(docs)} timestamped documents.")
        print(f"📦 Creating and saving FAISS embeddings ({AI_PROVIDER}) to disk...")
        vector_store = build_or_load_vector_store(video_id, docs, provider=AI_PROVIDER)

    print(f"🚀 Video is indexed and ready for questions via {AI_PROVIDER.upper()}!")
    print("\n💡 Commands:")
    print("   - Type your question and press Enter")
    print("   - Type '/summary' for an instant comprehensive summary")
    print("   - Type 'exit' or 'quit' to exit")
    print("-" * 60)

    rag = YouTubeRAG(vector_store, meta, provider=AI_PROVIDER)
    chat_history = []

    while True:
        try:
            query = input("\n💬 You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if query.lower() == "/summary":
                print("\n⏳ Generating executive summary...")
                summary = rag.generate_summary(full_transcript)
                print(f"\n{summary}\n")
                continue

            print("⏳ Thinking...")
            res = rag.ask(query, chat_history=chat_history)
            answer = res["answer"]
            sources = res["sources"]

            print(f"\n🤖 AI:\n{answer}\n")
            if sources:
                print("📌 Sources cited:")
                for s in sources:
                    print(f"   • [{s['timestamp']}] ({s['url']}) : {s['text'][:90]}...")

            chat_history.append((query, answer))
            if len(chat_history) > 5:
                chat_history.pop(0)

        except KeyboardInterrupt:
            print("\n👋 Exiting session.")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")

if __name__ == "__main__":
    main()
