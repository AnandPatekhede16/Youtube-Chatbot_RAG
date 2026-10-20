import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import REPURPOSE_DIR, AI_PROVIDER
from src.rag_engine import invoke_with_resilience

logger = logging.getLogger(__name__)

VIRAL_CLIPS_PROMPT = """You are a world-class viral short-form content producer (TikTok, YouTube Shorts, Instagram Reels) and video editor.
Analyze the following YouTube video transcript and identify the Top {num_clips} most viral, high-retention clip segments (30 to 90 seconds each).

Transcript:
{transcript_text}

Criteria for Viral Clips:
1. Pattern Interrupt or Hook: Must start with a bold statement, provocative question, or counter-intuitive claim.
2. Emotional or High Value: Provides an "aha!" moment, practical biohack, controversy, or relatable story.
3. Accurate Timestamps: Provide exact start and end times in MM:SS (e.g. "04:15" to "05:05").

Return ONLY a valid JSON array matching this exact schema (no markdown fences, no backticks, no extra text):
[
  {{
    "id": 1,
    "hook_title": "Why 99% Of People Fail At Morning Focus",
    "start_time": "04:15",
    "start_seconds": 255,
    "end_time": "05:05",
    "end_seconds": 305,
    "duration_seconds": 50,
    "virality_score": 95,
    "hook_type": "Counter-Intuitive Truth",
    "key_punchline": "Attention residue kills your deep work before you even open your laptop.",
    "why_it_works": "Attacks common habits with scientific backing, sparking high debate and shares in comments.",
    "recommended_caption": "The brutal reason your focus is ruined by 10 AM (and how to fix it) 👇 #focus #productivity #brainhacks"
  }}
]
"""

TWITTER_THREAD_PROMPT = """You are an elite ghostwriter who creates viral Twitter (X) threads for top founders and researchers.
Turn the following YouTube video transcript into a high-engagement 7 to 9 tweet thread.

Transcript:
{transcript_text}

Rules:
1. Tweet 1: Irresistible hook that promises high value and stops scrolling (mention speaker/topic).
2. Tweets 2 to 7: Distinct key lessons or breakthroughs with bullet points and exact video timestamp cues (e.g. "[14:20]").
3. Tweet 8: One-sentence golden rule summary.
4. Final Tweet: Call-to-Action to RT the first tweet and bookmark the thread.

Return ONLY a valid JSON object matching this exact schema (no markdown fences, no backticks):
{{
  "hook_headline": "Short description of the thread's core value",
  "tweets": [
    "Tweet 1 text...",
    "Tweet 2 text...",
    "Tweet 3 text..."
  ],
  "estimated_read_time": "2 mins",
  "recommended_hashtags": ["#Productivity", "#AI", "#Learning"]
}}
"""

LINKEDIN_POST_PROMPT = """You are a top 1% LinkedIn creator and thought leader.
Transform the following YouTube video transcript into a viral, high-performing LinkedIn post.

Transcript:
{transcript_text}

Formatting rules for LinkedIn:
1. 1-2 sentence hook line with white space below it (the "see more" click trigger).
2. Clean, punchy spacing (short 1-2 sentence paragraphs).
3. 4-6 bullet points covering the biggest insights and practical takeaways with timestamps.
4. Clear takeaway message for ambitious professionals.
5. An engaging question at the end to generate high comment velocity.
6. 3-5 relevant hashtags.

Return ONLY a valid JSON object matching this exact schema (no markdown fences, no backticks):
{{
  "post_title": "Catchy title for the post",
  "full_post_text": "Complete LinkedIn post formatted with spacing, bullet points, and hashtags...",
  "recommended_hook": "The opening 2 lines designed for maximum click-through rate",
  "call_to_action_question": "The question asked to readers at the end"
}}
"""

SEO_BLOG_PROMPT = """You are an expert SEO content strategist and executive tech blogger.
Write a comprehensive, publication-ready SEO blog post (1,200 to 1,500 words) based on the following YouTube video transcript and metadata.

Video Title: {video_title}
Creator/Speaker: {author}
Transcript:
{transcript_text}

Structure the blog post in clean Markdown:
# Catchy, Search-Optimized H1 Headline (Includes Primary Keyword)
**Meta Description**: Concise 155-character summary for Google search results.

## Introduction: The Core Challenge
Hook the reader, introduce the topic, and explain why this matters right now.

## 🔑 Key Breakthroughs & Core Pillars
(Include 3 to 4 detailed H2/H3 subheadings with deep explanations, bullet points, and timestamp citations like [MM:SS] to reference the video).

> 💡 **Key Quote from {author}**: Highlight an insightful quote from the video.

## 🛠️ Step-by-Step Practical Implementation Guide
Actionable checklist or framework the reader can immediately apply.

## ❓ Frequently Asked Questions (FAQ)
Provide 3 relevant FAQs with concise, direct answers suitable for Google featured snippets.

## 🎬 Conclusion & Watch the Full Discussion
Wrap up with an inspiring summary and invitation to explore the full conversation.
"""

def generate_viral_clips(
    video_id: str,
    transcript_text: str,
    num_clips: int = 4,
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Identifies top viral short-form moments with hooks, virality scores, and captions.
    Caches to disk in storage/repurpose/{video_id}_viral_clips.json.
    """
    cache_file = REPURPOSE_DIR / f"{video_id}_viral_clips.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception as e:
            logger.warning(f"Error reading viral clips cache: {e}")

    # Limit transcript to ~25,000 chars for API efficiency
    trimmed_text = transcript_text[:25000]

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(VIRAL_CLIPS_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        raw_output = invoke_with_resilience(
            build_chain,
            {"transcript_text": trimmed_text, "num_clips": num_clips},
            provider=provider,
            api_key=api_key
        )

        clean_json = raw_output.strip()
        clean_json = re.sub(r"^```(?:json)?", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        clips_data = json.loads(clean_json)
        if isinstance(clips_data, list):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(clips_data, f, ensure_ascii=False, indent=2)
            return clips_data
    except Exception as e:
        logger.error(f"Failed to generate viral clips: {e}")

    return []

def generate_social_thread(
    video_id: str,
    transcript_text: str,
    platform: str = "twitter",
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a viral Twitter thread or LinkedIn post.
    Caches to disk in storage/repurpose/{video_id}_{platform}_thread.json.
    """
    platform = platform.lower()
    cache_file = REPURPOSE_DIR / f"{video_id}_{platform}_thread.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception as e:
            logger.warning(f"Error reading {platform} thread cache: {e}")

    trimmed_text = transcript_text[:25000]
    prompt_str = TWITTER_THREAD_PROMPT if platform == "twitter" else LINKEDIN_POST_PROMPT

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(prompt_str)
        return prompt | llm | StrOutputParser()

    try:
        raw_output = invoke_with_resilience(
            build_chain,
            {"transcript_text": trimmed_text},
            provider=provider,
            api_key=api_key
        )

        clean_json = raw_output.strip()
        clean_json = re.sub(r"^```(?:json)?", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        thread_data = json.loads(clean_json)
        if isinstance(thread_data, dict):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(thread_data, f, ensure_ascii=False, indent=2)
            return thread_data
    except Exception as e:
        logger.error(f"Failed to generate {platform} thread: {e}")

    return {}

def generate_seo_blog(
    video_id: str,
    transcript_text: str,
    video_metadata: Dict[str, Any],
    provider: str = AI_PROVIDER,
    api_key: Optional[str] = None
) -> str:
    """
    Generates a full 1,200 - 1,500 word SEO-optimized blog post in Markdown format.
    Caches to disk in storage/repurpose/{video_id}_seo_blog.md.
    """
    cache_file = REPURPOSE_DIR / f"{video_id}_seo_blog.md"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    return content
        except Exception as e:
            logger.warning(f"Error reading SEO blog cache: {e}")

    trimmed_text = transcript_text[:25000]
    title = video_metadata.get("title", f"Video {video_id}")
    author = video_metadata.get("author", "the speaker")

    def build_chain(llm):
        prompt = ChatPromptTemplate.from_template(SEO_BLOG_PROMPT)
        return prompt | llm | StrOutputParser()

    try:
        blog_markdown = invoke_with_resilience(
            build_chain,
            {
                "video_title": title,
                "author": author,
                "transcript_text": trimmed_text
            },
            provider=provider,
            api_key=api_key
        )

        if blog_markdown:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(blog_markdown)
            return blog_markdown
    except Exception as e:
        logger.error(f"Failed to generate SEO blog: {e}")

    return ""
