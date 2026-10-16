import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import STUDY_DIR, AI_PROVIDER
from src.rag_engine import invoke_with_resilience

logger = logging.getLogger(__name__)

QUIZ_PROMPT = """You are an elite educational specialist and assessment designer.
Based on the following video transcript, create a high-yield {num_questions}-question Multiple Choice Knowledge Quiz to test deep comprehension.

Video Transcript:
{transcript_text}

Rules:
1. Each question must test a meaningful concept, breakthrough, or actionable takeaway discussed in the video.
2. Provide exactly 4 distinct options (A, B, C, D).
3. Specify the correct option index (0 for first, 1 for second, 2 for third, 3 for fourth).
4. Provide a clear, educational explanation citing why the answer is correct.
5. If available, mention an approximate timestamp (e.g. "12:45" or "N/A").

Return ONLY a valid JSON array matching this exact schema (no markdown formatting, no backticks, no extra text):
[
  {{
    "id": 1,
    "question": "What is the primary scientific reason multitasking degrades productivity?",
    "options": [
      "The brain consumes too much water during task switching",
      "Human brains cannot execute simultaneous cognitive tasks, creating attention residue",
      "Multitasking causes immediate physical muscle fatigue",
      "Only visual cortex processing is slowed down"
    ],
    "answer_index": 1,
    "explanation": "As explained in the dialogue, cognitive switching creates attention residue and drains glucose reserves.",
    "timestamp": "14:20"
  }}
]
"""

FLASHCARDS_PROMPT = """You are a master study coach.
Based on the following video transcript, generate {num_cards} high-impact concept study flashcards (Question/Prompt on Front, Clear Actionable Insight on Back).

Video Transcript:
{transcript_text}

Return ONLY a valid JSON array matching this exact schema (no markdown formatting, no backticks, no extra text):
[
  {{
    "id": 1,
    "front": "The 'Phone on Desk' Cognitive Drain Effect",
    "back": "Even when turned off and placed face-down, having a smartphone in your field of vision reduces active cognitive bandwidth because the prefrontal cortex actively expends energy to ignore it.",
    "tag": "Cognitive Science",
    "timestamp": "22:15"
  }}
]
"""

MINDMAP_PROMPT = """You are an expert system architect and visual knowledge cartographer.
Analyze the following video transcript and generate a comprehensive, highly readable Mermaid.js concept flowchart representing the central thesis, core sub-themes, and key actionable insights.

Video Transcript:
{transcript_text}

Rules:
1. Output ONLY valid Mermaid.js code starting with `flowchart TD` or `graph TD`.
2. Keep node IDs alphanumeric (e.g., A, B1, B2, C1) and always enclose text in double quotes inside brackets: A["Central Topic"] --> B["Core Insight"].
3. Do NOT include HTML tags inside labels.
4. Do NOT include backticks or markdown fences like ```mermaid - output ONLY the raw mermaid lines.
5. Build a structured 3-tier hierarchy: Central Thesis --> 3-4 Major Pillars --> Key Action Items.
"""

def _clean_json_text(raw_text: str) -> str:
    """Strip markdown code fences and whitespace from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def generate_quiz(
    video_id: str,
    transcript_text: str,
    num_questions: int = 5,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates or loads a cached multiple-choice knowledge quiz from video transcript.
    """
    cache_file = STUDY_DIR / f"{video_id}_quiz.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading quiz cache: {e}")

    # Limit text to ~25,000 characters for speed and economy
    sample_text = transcript_text[:25000]

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        raw_resp = invoke_with_resilience(
            build_chain,
            {"transcript_text": sample_text, "num_questions": num_questions},
            provider=provider,
            api_key=api_key
        )
        clean = _clean_json_text(raw_resp)
        data = json.loads(clean)
        if isinstance(data, list):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data
    except Exception as e:
        logger.error(f"Failed to generate quiz: {e}")

    return [
        {
            "id": 1,
            "question": "What is the key takeaway highlighted in this discussion?",
            "options": [
                "Focused execution and deep work yield higher results than continuous task switching",
                "Working without breaks is the optimal strategy",
                "Digital notifications have zero cognitive impact",
                "Routine task planning is counterproductive"
            ],
            "answer_index": 0,
            "explanation": "Focused deep work preserves attention bandwidth and avoids the severe productivity cost of constant task switching.",
            "timestamp": "05:00"
        }
    ]

def generate_flashcards(
    video_id: str,
    transcript_text: str,
    num_cards: int = 6,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates or loads cached concept flashcards from video transcript.
    """
    cache_file = STUDY_DIR / f"{video_id}_flashcards.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading flashcards cache: {e}")

    sample_text = transcript_text[:25000]

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(FLASHCARDS_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        raw_resp = invoke_with_resilience(
            build_chain,
            {"transcript_text": sample_text, "num_cards": num_cards},
            provider=provider,
            api_key=api_key
        )
        clean = _clean_json_text(raw_resp)
        data = json.loads(clean)
        if isinstance(data, list):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data
    except Exception as e:
        logger.error(f"Failed to generate flashcards: {e}")

    return [
        {
            "id": 1,
            "front": "Core Principle of Focused Attention",
            "back": "Sustained attention on a single priority minimizes cognitive friction and preserves brain glucose.",
            "tag": "Productivity",
            "timestamp": "04:30"
        }
    ]

def generate_mindmap(
    video_id: str,
    transcript_text: str,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> str:
    """
    Generates or loads a cached Mermaid.js concept flowchart.
    """
    cache_file = STUDY_DIR / f"{video_id}_mindmap.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("mermaid_code", "")
        except Exception as e:
            logger.warning(f"Error reading mindmap cache: {e}")

    sample_text = transcript_text[:25000]

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(MINDMAP_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        raw_resp = invoke_with_resilience(
            build_chain,
            {"transcript_text": sample_text},
            provider=provider,
            api_key=api_key
        )
        cleaned = raw_resp.strip()
        # Remove markdown fences if present
        if cleaned.startswith("```mermaid"):
            cleaned = cleaned[10:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"mermaid_code": cleaned}, f, indent=2, ensure_ascii=False)
        return cleaned
    except Exception as e:
        logger.error(f"Failed to generate mindmap: {e}")

    return """graph TD
    A["Video Core Theme"] --> B1["Primary Principle"]
    A --> B2["Tactical Strategy"]
    A --> B3["Key Breakthrough"]
    B1 --> C1["Preserve Attention Reserves"]
    B2 --> C2["Eliminate Ambient Distractions"]
    B3 --> C3["Sustainable Daily Cadence"]"""

def generate_study_notes_markdown(
    video_meta: Dict[str, Any],
    summary_text: Optional[str],
    quiz_data: Optional[List[Dict[str, Any]]],
    flashcards_data: Optional[List[Dict[str, Any]]],
    mindmap_code: Optional[str],
    comments_summary: Optional[str] = None
) -> str:
    """
    Assembles a complete, structured, printable study guide in Markdown.
    """
    title = video_meta.get("title", "YouTube Video Study Notes")
    author = video_meta.get("author", "Creator")
    video_id = video_meta.get("video_id", "")
    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else "#"

    md = []
    md.append(f"# 🎓 Study Guide & Executive Brief: {title}\n")
    md.append(f"**Presenter / Channel:** {author} | **Source Video:** [{url}]({url})\n")
    md.append("---\n")

    if summary_text:
        md.append("## 📋 Executive Summary & Key Highlights\n")
        md.append(f"{summary_text}\n\n")

    if comments_summary:
        md.append("## 🗣️ Audience Consensus & Community Takeaways\n")
        md.append(f"{comments_summary}\n\n")

    if flashcards_data:
        md.append("## 🃏 Core Concept Flashcards\n")
        for i, card in enumerate(flashcards_data, 1):
            tag = card.get("tag", "General")
            ts = card.get("timestamp", "N/A")
            md.append(f"### Card {i}: {card.get('front')} `[{tag}]` (⏱️ {ts})")
            md.append(f"> **Insight:** {card.get('back')}\n")

    if quiz_data:
        md.append("## 🎯 Comprehension Quiz & Solutions\n")
        for i, q in enumerate(quiz_data, 1):
            md.append(f"**Q{i}. {q.get('question')}** (⏱️ {q.get('timestamp', 'N/A')})")
            ans_idx = q.get("answer_index", 0)
            for j, opt in enumerate(q.get("options", [])):
                prefix = "(Correct) " if j == ans_idx else ""
                md.append(f"- [{ 'x' if j == ans_idx else ' ' }] {prefix}{opt}")
            md.append(f"*Explanation:* {q.get('explanation')}\n")

    if mindmap_code:
        md.append("## 🗺️ Concept Mind Map (Mermaid)\n")
        md.append("```mermaid")
        md.append(mindmap_code)
        md.append("```\n")

    md.append("\n---\n*Generated by TubeMind AI — Retrieval-Augmented Generation & Video Intelligence.*")
    return "\n".join(md)
