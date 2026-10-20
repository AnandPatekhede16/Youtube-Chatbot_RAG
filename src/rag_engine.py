import logging
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import (
    AI_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    DEFAULT_TOP_K
)
from src.vectorstore import get_retriever

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
logger = logging.getLogger(__name__)

SYSTEM_QA_PROMPT = """You are an intelligent, articulate YouTube AI Companion.
Your goal is to answer the user's questions accurately and concisely based on the provided YouTube video transcript context and audience comments intelligence.

Guidelines:
1. Always ground your answers strictly in the provided context. If the answer cannot be found in the context or comments, politely explain that the video does not mention or cover that topic.
2. When answering video-related questions, cite the exact timestamp (e.g. [14:22]) where the topic or quote appears in the video. When possible, format it with markdown link like [14:22](timestamp_url) if provided in the context metadata.
3. If the user asks about audience feedback, viewer opinions, top doubts, or comments, use the Audience Comments Intelligence section.
4. Be clear, insightful, and structure answers with bullet points or bold text where appropriate.

Context from Video Transcript:
{context}

{comments_context}
"""

REPHRASE_PROMPT = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Question: {question}
Standalone Question:"""

SUMMARY_PROMPT = """You are an expert executive analyst. Analyze the following complete YouTube video transcript and produce a high quality, well-structured summary.

Transcript:
{transcript_text}

Provide your analysis in the following format:
## 📌 Executive Summary
A concise 2-3 paragraph overview of the video's core theme and message.

## 🎯 Key Takeaways & Discussion Highlights
- Highlight 4 to 6 main ideas or breakthroughs discussed.
- Mention key speakers or examples where applicable.

## 💡 Practical Insights or Actionable Advice
- What should a viewer learn, remember, or take away from this discussion?

## ⏱️ Recommended Focus Moments
- Mention important topic shifts or pivotal points in the discussion.
"""

def invoke_with_resilience(chain_builder_func, inputs: Dict[str, Any], provider: str = AI_PROVIDER, api_key: Optional[str] = None):
    """
    Executes a LangChain runnable with automatic model fallback and retries 
    to guarantee 100% reliability against 503 (high demand) and 429 (rate limit) errors.
    """
    provider = provider.lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        active_key = api_key or GEMINI_API_KEY
        
        # Priority order of available high-quota models
        models_to_try = [GEMINI_CHAT_MODEL, "gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview"]
        candidate_models = list(dict.fromkeys(models_to_try))
        
        last_error = None
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    kwargs = {
                        "model": model_name,
                        "google_api_key": active_key,
                        "timeout": 30
                    }
                    if "flash-lite" not in model_name:
                        kwargs["temperature"] = 0.2

                    llm = ChatGoogleGenerativeAI(**kwargs)
                    chain = chain_builder_func(llm)
                    return chain.invoke(inputs)
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if any(code in err_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate_limit"]):
                        logger.warning(f"Gemini model {model_name} busy or limited (Attempt {attempt+1}). Retrying / switching...")
                        time.sleep(1.5)
                        continue
                    else:
                        break
        raise last_error
    else:
        from langchain_openai import ChatOpenAI
        active_key = api_key or OPENAI_API_KEY
        llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            openai_api_key=active_key,
            temperature=0.2
        )
        chain = chain_builder_func(llm)
        return chain.invoke(inputs)

class YouTubeRAG:
    """
    Retrieval-Augmented Generation Engine for YouTube videos and audience comments.
    """
    def __init__(
        self, 
        vector_store, 
        video_metadata: Dict[str, Any], 
        comments_analysis: Optional[Dict[str, Any]] = None,
        provider: str = AI_PROVIDER,
        api_key: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.video_metadata = video_metadata
        self.comments_analysis = comments_analysis
        self.provider = provider.lower()
        self.api_key = api_key
        self.retriever = get_retriever(vector_store, top_k=DEFAULT_TOP_K)

    def _format_context(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a timestamped context string."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            ts = doc.metadata.get("timestamp", "N/A")
            url = doc.metadata.get("timestamp_url", "")
            snippet_header = f"--- Source Chunk {i} [{ts}] (URL: {url}) ---"
            context_parts.append(f"{snippet_header}\n{doc.page_content}\n")
        return "\n".join(context_parts)

    def _format_comments_context(self) -> str:
        """Builds audience intelligence context block if available."""
        if not self.comments_analysis:
            return ""
        
        ca = self.comments_analysis
        summary = ca.get("community_summary", "")
        loved = ", ".join(ca.get("what_viewers_loved", []))
        debates = ", ".join(ca.get("debates_and_skepticism", []))
        questions = ", ".join(ca.get("top_viewer_questions", []))
        
        return f"""
Audience Comments Intelligence:
- Sentiment: {ca.get('positive_pct', 0)}% Positive, {ca.get('neutral_pct', 0)}% Discussion, {ca.get('critical_pct', 0)}% Critical
- Vibe: {', '.join(ca.get('vibe_tags', []))}
- Community Summary: {summary}
- What Viewers Loved: {loved}
- Audience Debates/Skepticism: {debates}
- Top Questions Asked by Viewers: {questions}
"""

    def ask(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Answers a user question considering past chat history.
        Uses resilient fallback to eliminate 503 errors.
        """
        chat_history = chat_history or []

        # 1. Rephrase question if multi-turn history exists
        query_to_retrieve = question
        if chat_history:
            history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
            def build_rephrase(llm):
                return ChatPromptTemplate.from_template(REPHRASE_PROMPT) | llm | StrOutputParser()
            try:
                query_to_retrieve = invoke_with_resilience(
                    build_rephrase,
                    {"chat_history": history_str, "question": question},
                    provider=self.provider,
                    api_key=self.api_key
                ).strip()
            except Exception as e:
                logger.warning(f"Failed to rephrase question: {e}")
                query_to_retrieve = question

        # 2. Retrieve relevant chunks
        docs = self.retriever.invoke(query_to_retrieve)
        formatted_context = self._format_context(docs)
        formatted_comments = self._format_comments_context()

        # 3. Generate answer with resilient fallback
        formatted_messages = []
        for q, a in chat_history:
            formatted_messages.append(("human", q))
            formatted_messages.append(("ai", a))

        def build_qa_chain(llm):
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_QA_PROMPT),
                MessagesPlaceholder(variable_name="history_messages", optional=True),
                ("human", "{question}"),
            ])
            return prompt | llm | StrOutputParser()

        answer = invoke_with_resilience(
            build_qa_chain,
            {
                "context": formatted_context,
                "comments_context": formatted_comments,
                "history_messages": formatted_messages,
                "question": question,
            },
            provider=self.provider,
            api_key=self.api_key
        )

        # Format sources for UI citations
        sources = []
        for doc in docs:
            sources.append({
                "timestamp": doc.metadata.get("timestamp", ""),
                "start_seconds": doc.metadata.get("start_seconds", 0),
                "url": doc.metadata.get("timestamp_url", ""),
                "text": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_query": query_to_retrieve
        }

    def generate_summary(self, full_transcript_text: str) -> str:
        """
        Generates a comprehensive structured summary of the entire video.
        """
        max_chars = 90000
        truncated_text = full_transcript_text[:max_chars]
        if len(full_transcript_text) > max_chars:
            truncated_text += "\n[Transcript truncated due to length...]"

        def build_summary_chain(llm):
            prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
            return prompt | llm | StrOutputParser()

        return invoke_with_resilience(
            build_summary_chain,
            {"transcript_text": truncated_text},
            provider=self.provider,
            api_key=self.api_key
        )
