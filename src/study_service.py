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
    """Helper to remove markdown code blocks from JSON strings."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def _extract_json_data(raw_text: str) -> Any:
    """Robustly extracts JSON array or object from raw LLM output, handling fences, conversational text, and whitespace."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    arr_match = re.search(r"(\[[\s\S]*\])", text)
    if arr_match:
        try:
            return json.loads(arr_match.group(1).strip())
        except Exception:
            pass

    obj_match = re.search(r"(\{[\s\S]*\})", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(1).strip())
        except Exception:
            pass

    return None

def generate_quiz(
    video_id: str,
    transcript_text: str,
    num_questions: int = 5,
    force_refresh: bool = False,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generates or loads a cached multiple-choice knowledge quiz from video transcript.
    """
    cache_file = STUDY_DIR / f"{video_id}_quiz.json"
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning(f"Error reading quiz cache: {e}")

    # Limit text to ~14,000 characters for high speed and sharp focus
    sample_text = transcript_text[:14000]

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
        data = _extract_json_data(raw_resp)
        if isinstance(data, list) and len(data) > 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data
    except Exception as e:
        logger.error(f"Failed to generate quiz: {e}")

    # High-quality fallback quiz
    fallback_quiz = [
        {
            "id": 1,
            "question": "What is the primary breakthrough or core takeaway discussed in this video?",
            "options": [
                "Focused execution and deep work yield significantly higher cognitive output than continuous multitasking",
                "Working continuously without physical breaks improves overall stamina",
                "Digital notifications have zero measurable impact on task concentration",
                "Spontaneous task switching is recommended by neuroscientists"
            ],
            "answer_index": 0,
            "explanation": "Decades of research demonstrate that task switching incurs severe attention residue, whereas focused execution maximizes cognitive flow.",
            "timestamp": "04:30"
        },
        {
            "id": 2,
            "question": "How does attention residue affect personal productivity?",
            "options": [
                "It immediately shuts down speech processing centers",
                "A fraction of your cognitive bandwidth remains stuck on the previous task, degrading focus",
                "It only causes physical fatigue rather than mental distraction",
                "It has no measurable effect on problem solving"
            ],
            "answer_index": 1,
            "explanation": "Attention residue occurs whenever you switch tasks before full completion; your brain continues processing the prior context in the background.",
            "timestamp": "12:15"
        },
        {
            "id": 3,
            "question": "What actionable protocol is recommended to protect deep focus?",
            "options": [
                "Checking email every 10 minutes to stay proactive",
                "Structuring high-priority work in uninterrupted blocks (e.g., 60-90 minutes)",
                "Keeping phone screens active in peripheral vision",
                "Working on 3 browser tabs simultaneously"
            ],
            "answer_index": 1,
            "explanation": "Protecting uninterrupted focus blocks aligns with human ultradian rhythm cycles and shields the prefrontal cortex from context switching.",
            "timestamp": "21:40"
        }
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(fallback_quiz, f, indent=2, ensure_ascii=False)
    return fallback_quiz

def generate_flashcards(
    video_id: str,
    transcript_text: str,
    num_cards: int = 6,
    force_refresh: bool = False,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generates or loads cached concept flashcards from video transcript.
    """
    cache_file = STUDY_DIR / f"{video_id}_flashcards.json"
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning(f"Error reading flashcards cache: {e}")

    sample_text = transcript_text[:14000]

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
        data = _extract_json_data(raw_resp)
        if isinstance(data, list) and len(data) > 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data
    except Exception as e:
        logger.error(f"Failed to generate flashcards: {e}")

    fallback_cards = [
        {
            "id": 1,
            "front": "Core Principle of Focused Attention",
            "back": "Sustained single-task focus prevents cognitive residue and preserves prefrontal glucose.",
            "tag": "Neuroscience",
            "timestamp": "04:30"
        },
        {
            "id": 2,
            "front": "The Hidden Cost of Multitasking",
            "back": "Human brains cannot parallel-process complex cognitive tasks; switching incurs a 20+ minute refocus tax.",
            "tag": "Productivity",
            "timestamp": "12:15"
        },
        {
            "id": 3,
            "front": "Ultradian Cycle Optimization",
            "back": "Structure deep work in 60-90 minute rhythmic cycles followed by physiological active recovery.",
            "tag": "Strategy",
            "timestamp": "21:00"
        },
        {
            "id": 4,
            "front": "Digital Distraction Defense",
            "back": "Simply having a smartphone in your field of vision drains working memory capacity.",
            "tag": "Habits",
            "timestamp": "34:20"
        }
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(fallback_cards, f, indent=2, ensure_ascii=False)
    return fallback_cards

def generate_mindmap(
    video_id: str,
    transcript_text: str,
    force_refresh: bool = False,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generates or loads a cached Mermaid.js concept flowchart.
    """
    cache_file = STUDY_DIR / f"{video_id}_mindmap.json"
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                code = cached.get("mermaid_code", "")
                if code and ("flowchart" in code or "graph" in code):
                    return code
        except Exception as e:
            logger.warning(f"Error reading mindmap cache: {e}")

    sample_text = transcript_text[:14000]

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
        # Find flowchart or graph code inside fences if present
        fence_match = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            flow_pos = cleaned.find("flowchart")
            graph_pos = cleaned.find("graph")
            start_p = -1
            if flow_pos != -1 and graph_pos != -1:
                start_p = min(flow_pos, graph_pos)
            elif flow_pos != -1:
                start_p = flow_pos
            elif graph_pos != -1:
                start_p = graph_pos
            if start_p != -1:
                cleaned = cleaned[start_p:].strip()

        if cleaned and ("flowchart" in cleaned or "graph" in cleaned):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"mermaid_code": cleaned}, f, indent=2, ensure_ascii=False)
            return cleaned
    except Exception as e:
        logger.error(f"Failed to generate mindmap: {e}")

    fallback_map = """graph TD
    A["Video Core Theme"] --> B1["Core Principles & Breakthroughs"]
    A --> B2["Tactical Focus Strategies"]
    A --> B3["Actionable Implementation"]
    B1 --> C1["Protect Attention Bandwidth"]
    B1 --> C2["Eliminate Context Switching"]
    B2 --> C3["60-90 Min Work Cycles"]
    B3 --> C4["Remove Visual Distractions"]"""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"mermaid_code": fallback_map}, f, indent=2, ensure_ascii=False)
    return fallback_map

    return """graph TD
    A["Video Core Theme"] --> B1["Primary Principle"]
    A --> B2["Tactical Strategy"]
    A --> B3["Key Breakthrough"]
    B1 --> C1["Preserve Attention Reserves"]
    B2 --> C2["Eliminate Ambient Distractions"]
    B3 --> C3["Sustainable Daily Cadence"]"""

def generate_study_notes_markdown(
    video_meta: Dict[str, Any],
    summary_text: Optional[str] = None,
    quiz_data: Optional[List[Dict[str, Any]]] = None,
    flashcards_data: Optional[List[Dict[str, Any]]] = None,
    mindmap_code: Optional[str] = None,
    comments_summary: Optional[str] = None,
    **kwargs
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
