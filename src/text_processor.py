from typing import List, Dict, Any
from langchain_core.documents import Document
from src.youtube_service import format_timestamp

class TimestampAwareChunker:
    """
    Groups raw transcript snippets into semantic chunks while strictly
    preserving starting/ending timestamps and generating deep-link URLs.
    """
    def __init__(self, target_chunk_size: int = 1000, overlap_snippets: int = 2):
        self.target_chunk_size = target_chunk_size
        self.overlap_snippets = overlap_snippets

    def create_documents(self, snippets: List[Dict[str, Any]], video_id: str) -> List[Document]:
        """
        Converts raw transcript snippets into a list of LangChain Document objects
        with rich metadata including timestamps and YouTube time-stamped links.
        """
        if not snippets:
            return []

        documents: List[Document] = []
        n = len(snippets)
        i = 0

        while i < n:
            current_texts: List[str] = []
            chunk_char_count = 0
            start_time = float(snippets[i].get("start", 0.0))
            end_time = start_time
            last_idx = i

            # Accumulate snippets until target size is reached
            while last_idx < n and chunk_char_count < self.target_chunk_size:
                snip = snippets[last_idx]
                text = snip.get("text", "").strip()
                if text:
                    current_texts.append(text)
                    chunk_char_count += len(text) + 1
                    s_start = float(snip.get("start", 0.0))
                    s_dur = float(snip.get("duration", 0.0))
                    end_time = max(end_time, s_start + s_dur)
                last_idx += 1

            chunk_text = " ".join(current_texts)
            timestamp_str = format_timestamp(start_time)
            int_start_sec = int(start_time)
            timestamp_url = f"https://youtu.be/{video_id}?t={int_start_sec}"

            # Format content with timestamp banner so the LLM explicitly sees it during retrieval
            page_content = f"[{timestamp_str}] {chunk_text}"

            metadata = {
                "video_id": video_id,
                "start_seconds": start_time,
                "end_seconds": end_time,
                "timestamp": timestamp_str,
                "timestamp_url": timestamp_url,
                "snippet_count": last_idx - i,
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

            # Move forward with snippet overlap
            if last_idx >= n:
                break
            step = max(1, (last_idx - i) - self.overlap_snippets)
            i += step

        return documents

def get_full_transcript_text(snippets: List[Dict[str, Any]]) -> str:
    """Combines all snippets into a single continuous plain text string."""
    return " ".join(s.get("text", "").strip() for s in snippets if s.get("text"))
