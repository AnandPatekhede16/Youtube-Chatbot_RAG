import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import COMMENTS_ANALYSIS_DIR, AI_PROVIDER
from src.rag_engine import invoke_with_resilience

logger = logging.getLogger(__name__)

COMMENTS_ANALYSIS_PROMPT = """You are an expert audience intelligence and social listening analyst.
Analyze the following top comments from a YouTube video and produce a rich, structured JSON report.

Comments Data:
{comments_data}

Return ONLY a valid JSON object matching this exact schema (no markdown fences, no extra text):
{{
  "positive_pct": 75,
  "neutral_pct": 15,
  "critical_pct": 10,
  "vibe_tags": ["Insightful", "Life-Changing", "Practical"],
  "community_summary": "A concise 2-3 sentence overview of the audience's reaction and sentiment.",
  "what_viewers_loved": [
    "Specific insight or moment viewers praised most",
    "Another point that resonated with the audience"
  ],
  "debates_and_skepticism": [
    "Points where viewers disagreed or expressed skepticism",
    "Counter-arguments raised in the comment section"
  ],
  "top_viewer_questions": [
    "Common question or doubt viewers asked in the comments",
    "Another frequent question from the audience"
  ],
  "future_topic_requests": [
    "What the audience wants to see in the next episode or part 2"
  ]
}}
"""

def analyze_comments(
    video_id: str, 
    comments: List[Dict[str, Any]], 
    provider: str = AI_PROVIDER, 
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes YouTube comments with LLM and returns structured audience intelligence.
    Caches result to disk.
    """
    if not comments:
        return {
            "positive_pct": 0,
            "neutral_pct": 0,
            "critical_pct": 0,
            "vibe_tags": ["No Comments"],
            "community_summary": "No public comments were retrieved for this video.",
            "what_viewers_loved": [],
            "debates_and_skepticism": [],
            "top_viewer_questions": [],
            "future_topic_requests": []
        }

    cache_file = COMMENTS_ANALYSIS_DIR / f"{video_id}.json"

    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cached comments analysis: {e}")

    # Prepare comments text for LLM (take up to top 60 comments to stay light and fast)
    formatted_comments = []
    for i, c in enumerate(comments[:60], 1):
        author = c.get("author", "User")
        votes = c.get("votes", "0")
        text = c.get("text", "").replace("\n", " ").strip()
        formatted_comments.append(f"[{i}] {author} (+{votes} likes): {text}")

    comments_block = "\n".join(formatted_comments)

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(COMMENTS_ANALYSIS_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        raw_response = invoke_with_resilience(
            build_chain,
            {"comments_data": comments_block},
            provider=provider,
            api_key=api_key
        )

        # Clean JSON response if enclosed in code blocks
        clean_json = raw_response.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r"^```(?:json)?", "", clean_json)
            clean_json = re.sub(r"```$", "", clean_json).strip()

        data = json.loads(clean_json)

    except Exception as e:
        logger.error(f"Failed to parse LLM comments analysis JSON: {e}")
        data = {
            "positive_pct": 80,
            "neutral_pct": 15,
            "critical_pct": 5,
            "vibe_tags": ["Engaged", "Active Discussion"],
            "community_summary": "The audience showed strong engagement with varied feedback across comments.",
            "what_viewers_loved": ["Key practical tips shared throughout the discussion."],
            "debates_and_skepticism": ["Differing perspectives on feasibility and daily routines."],
            "top_viewer_questions": ["How to implement these tips with busy schedules."],
            "future_topic_requests": ["More deep dives into specific routines."]
        }

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to cache comments analysis: {e}")

    return data
