import streamlit as st
import time
import re
from typing import List, Dict, Any, Optional

from src.config import (
    validate_environment, 
    AI_PROVIDER, 
    GEMINI_API_KEY, 
    OPENAI_API_KEY
)
from src.youtube_service import (
    extract_video_id,
    fetch_video_metadata,
    get_transcript_snippets,
    format_timestamp
)
from src.comments_service import fetch_video_comments
from src.comments_analyzer import analyze_comments
from src.text_processor import TimestampAwareChunker, get_full_transcript_text
from src.vectorstore import build_or_load_vector_store, is_video_indexed
from src.rag_engine import YouTubeRAG

# 1. Page Configuration
st.set_page_config(
    page_title="TubeMind AI | Video & Audience Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom CSS: Hiding Streamlit Chrome & Realistic Website Look
st.markdown("""
<style>
    /* Hide Streamlit default hamburger menu, deploy button, footer & status decoration */
    #MainMenu,
    [data-testid="stMainMenu"],
    .stDeployButton,
    [data-testid="stDeployButton"],
    footer,
    div[data-testid="stDecoration"],
    [data-testid="stToolbarActions"],
    [data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* Top header bar transparent and unhindered */
    header[data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
    }

    /* MAKE THE EXPAND SIDEBAR BUTTON SUPER PROMINENT AND BEAUTIFUL (Visible when collapsed) */
    [data-testid="stExpandSidebarButton"],
    button[aria-label="Expand sidebar"],
    button[aria-label="Open sidebar"],
    button[data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #ff1e56 !important;
        color: #ffffff !important;
        border: 2px solid #ff4b72 !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        box-shadow: 0 4px 14px rgba(255, 30, 86, 0.45) !important;
        cursor: pointer !important;
        z-index: 999999 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    button[aria-label="Expand sidebar"]:hover,
    button[aria-label="Open sidebar"]:hover {
        background: #e01548 !important;
        transform: scale(1.08) !important;
        box-shadow: 0 6px 20px rgba(255, 30, 86, 0.6) !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    button[aria-label="Expand sidebar"] svg,
    button[aria-label="Open sidebar"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    /* Style the Sidebar Collapse Button cleanly */
    [data-testid="stSidebarCollapseButton"] button,
    button[aria-label="Collapse sidebar"],
    button[aria-label="Close sidebar"] {
        visibility: visible !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    button[aria-label="Collapse sidebar"]:hover {
        background: rgba(255, 30, 86, 0.15) !important;
        color: #ff1e56 !important;
    }

    /* App container spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Brand Header */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF1E56, #FF9900);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .brand-tagline {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    /* Theme-Adaptive Metric Cards */
    .metric-card {
        background: rgba(125, 125, 125, 0.08);
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        color: inherit;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        opacity: 0.75;
    }

    /* Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-green { background: rgba(34, 197, 94, 0.18); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-blue { background: rgba(59, 130, 246, 0.18); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.18); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.18); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Comment Card Style */
    .comment-card {
        background: rgba(125, 125, 125, 0.06);
        border: 1px solid rgba(125, 125, 125, 0.18);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .comment-author {
        font-weight: 600;
        color: inherit;
        font-size: 0.92rem;
    }
    .comment-meta {
        font-size: 0.78rem;
        opacity: 0.65;
        margin-bottom: 6px;
    }
    .comment-body {
        font-size: 0.90rem;
        line-height: 1.45;
        opacity: 0.92;
    }

    /* Timestamp tag */
    .ts-tag {
        display: inline-block;
        background: rgba(255, 30, 86, 0.15);
        color: #ff1e56;
        border: 1px solid rgba(255, 30, 86, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        text-decoration: none;
        margin-right: 4px;
        transition: background 0.2s ease;
    }
    .ts-tag:hover {
        background: rgba(255, 30, 86, 0.3);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_meta" not in st.session_state:
    st.session_state.video_meta = None
if "snippets" not in st.session_state:
    st.session_state.snippets = None
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = None
if "comments" not in st.session_state:
    st.session_state.comments = []
if "comments_analysis" not in st.session_state:
    st.session_state.comments_analysis = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# 4. Sidebar: Clean, Sleek SaaS Layout
with st.sidebar:
    st.markdown("### 🎬 TubeMind AI")
    st.caption("Video & Audience Social Listening Platform")
    st.markdown("---")

    st.markdown("#### 🔗 Load YouTube Video")
    url_input = st.text_input(
        "Paste YouTube URL or Video ID",
        value="https://www.youtube.com/watch?v=4Vz6L8B73i4",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    fetch_comments_toggle = st.toggle("Fetch & Analyze Audience Comments", value=True)

    load_btn = st.button("🚀 Analyze Video & Audience", use_container_width=True, type="primary")

    # Collapsible Settings tucked cleanly out of the main view
    with st.expander("⚙️ AI Model & API Configuration", expanded=False):
        provider_choice = st.selectbox(
            "AI Provider",
            options=["gemini", "openai"],
            index=0 if AI_PROVIDER == "gemini" else 1,
            format_func=lambda x: "✨ Google Gemini (Active & Free)" if x == "gemini" else "🤖 OpenAI (GPT-4o-mini)"
        )
        if provider_choice == "gemini":
            active_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
        else:
            active_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")

    # Processing Logic
    if load_btn:
        vid = extract_video_id(url_input)
        if not vid:
            st.error("❌ Please provide a valid YouTube URL or 11-character video ID.")
        elif not active_key or active_key.startswith("your_"):
            st.error(f"⚠️ Please enter a valid {provider_choice.title()} API key in the settings.")
        else:
            with st.spinner("Processing video transcript, index, and audience comments..."):
                try:
                    meta = fetch_video_metadata(vid)
                    snippets = get_transcript_snippets(vid)
                    full_txt = get_full_transcript_text(snippets)

                    # Build or load vector store
                    if is_video_indexed(vid, provider=provider_choice):
                        vector_store = build_or_load_vector_store(
                            vid, 
                            provider=provider_choice, 
                            api_key=active_key
                        )
                    else:
                        chunker = TimestampAwareChunker(target_chunk_size=1600, overlap_snippets=2)
                        docs = chunker.create_documents(snippets, vid)
                        vector_store = build_or_load_vector_store(
                            vid, 
                            docs, 
                            provider=provider_choice, 
                            api_key=active_key
                        )

                    # Comments Handling
                    comments_data = []
                    analysis_data = None
                    if fetch_comments_toggle:
                        comments_data = fetch_video_comments(vid, max_comments=50)
                        if comments_data:
                            analysis_data = analyze_comments(
                                vid, 
                                comments_data, 
                                provider=provider_choice, 
                                api_key=active_key
                            )

                    # Update State
                    st.session_state.video_id = vid
                    st.session_state.video_meta = meta
                    st.session_state.snippets = snippets
                    st.session_state.full_transcript = full_txt
                    st.session_state.comments = comments_data
                    st.session_state.comments_analysis = analysis_data
                    st.session_state.rag_engine = YouTubeRAG(
                        vector_store, 
                        meta, 
                        comments_analysis=analysis_data,
                        provider=provider_choice, 
                        api_key=active_key
                    )
                    st.session_state.chat_history = []
                    st.session_state.summary = None

                    st.success("✅ Analysis ready!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Video Card in Sidebar if loaded
    if st.session_state.video_id and st.session_state.video_meta:
        st.markdown("---")
        meta = st.session_state.video_meta
        st.markdown(f"**{meta.get('title', 'Video')}**")
        st.caption(f"👤 Channel: {meta.get('author', 'Unknown')}")

        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

        is_cached = is_video_indexed(st.session_state.video_id, provider=AI_PROVIDER)
        st.markdown(
            "<span class='badge-pill badge-green'>⚡ Index Cached on Disk</span>" 
            if is_cached else "<span class='badge-pill badge-blue'>✨ Newly Indexed</span>", 
            unsafe_allow_html=True
        )
        if st.session_state.comments:
            st.markdown(f"<span class='badge-pill badge-purple'>💬 {len(st.session_state.comments)} Comments Analyzed</span>", unsafe_allow_html=True)

# 5. Main Content Area
st.markdown("<div class='brand-title'>TubeMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-tagline'>Deep Retrieval-Augmented Generation for YouTube Videos & Real-Time Audience Intelligence.</div>", unsafe_allow_html=True)

if not st.session_state.rag_engine:
    # Landing View
    st.info("👈 Enter a YouTube URL in the sidebar and click **'Analyze Video & Audience'** to begin!")
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Grounded Video Q&A</div>
            <div class='metric-val'>⏱️ Deep Links</div>
            <p style='color:#64748b; font-size:0.88rem; margin-top:8px;'>
                Every AI answer is strictly grounded in the transcript and provides exact clickable timestamp citations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Audience Social Listening</div>
            <div class='metric-val'>🗣️ Sentiment & Debates</div>
            <p style='color:#64748b; font-size:0.88rem; margin-top:8px;'>
                Scans top comments for positive sentiment, community debates, crowdsourced timestamps, and audience doubts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with colC:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Executive Summaries</div>
            <div class='metric-val'>📋 1-Click Digest</div>
            <p style='color:#64748b; font-size:0.88rem; margin-top:8px;'>
                Generates a structured executive digest of 2-hour podcasts with key highlights and action items.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # 6. Fallback Notification if Subtitles Disabled
    if st.session_state.snippets and any(s.get("is_chapter") for s in st.session_state.snippets):
        ch_count = len([s for s in st.session_state.snippets if s.get("is_chapter")])
        st.info(f"ℹ️ **Note:** Speech subtitles were disabled by the creator for this video. TubeMind AI automatically recovered **{ch_count} Video Chapters & Timestamps** plus **Audience Comments** so complete Q&A, chapter navigation, and social listening work seamlessly!")

    # 7. Main Feature Tabs
    tab_chat, tab_comments, tab_summary, tab_transcript = st.tabs([
        "💬 Video & Audience Q&A", 
        "🗣️ Audience & Comments Hub", 
        "📋 Executive Summary", 
        "📜 Searchable Transcript"
    ])

    # ==========================================
    # TAB 1: INTERACTIVE CHAT
    # ==========================================
    with tab_chat:
        st.caption(f"Actively exploring: **{st.session_state.video_meta.get('title')}**")

        # Suggested Questions
        st.markdown("**💡 Quick Test Prompts:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        sample_q = None
        with col_q1:
            if st.button("🧠 Multitasking vs Task Switching?", use_container_width=True):
                sample_q = "What does she say about multitasking versus task switching?"
        with col_q2:
            if st.button("📱 Why Phone on Table Lowers IQ?", use_container_width=True):
                sample_q = "Why does having a phone visible on the table decrease IQ and focus?"
        with col_q3:
            if st.button("🗣️ What did viewers say in comments?", use_container_width=True):
                sample_q = "What are viewers in the comments saying about this discussion?"

        # Display Chat History
        for item in st.session_state.chat_history:
            role = item["role"]
            content = item["content"]
            sources = item.get("sources", [])

            with st.chat_message(role):
                st.markdown(content)
                if sources:
                    with st.expander("📌 View Timestamp Citations & Source Moments", expanded=False):
                        for s in sources:
                            ts = s.get("timestamp", "N/A")
                            url = s.get("url", "#")
                            snippet_text = s.get("text", "")
                            st.markdown(f"**[{ts}]({url})** — *\"{snippet_text}\"*")

        # Chat Input
        user_query = st.chat_input("Ask any question about the video or audience comments...")
        prompt_to_send = sample_q or user_query

        if prompt_to_send:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_send})
            with st.chat_message("user"):
                st.markdown(prompt_to_send)

            with st.chat_message("assistant"):
                with st.spinner("Searching transcript cues & audience insights..."):
                    formatted_hist = [
                        (m["content"], m.get("content", "")) 
                        for m in st.session_state.chat_history[:-1] 
                        if m["role"] == "user"
                    ]
                    
                    try:
                        result = st.session_state.rag_engine.ask(
                            prompt_to_send, 
                            chat_history=formatted_hist
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)
                        if sources:
                            with st.expander("📌 View Timestamp Citations & Source Moments", expanded=False):
                                for s in sources:
                                    ts = s.get("timestamp", "N/A")
                                    url = s.get("url", "#")
                                    snippet_text = s.get("text", "")
                                    st.markdown(f"**[{ts}]({url})** — *\"{snippet_text}\"*")

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        err_msg = f"❌ Error generating response: {str(e)}"
                        st.error(err_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": err_msg,
                            "sources": []
                        })

    # ==========================================
    # TAB 2: AUDIENCE & COMMENTS HUB
    # ==========================================
    with tab_comments:
        st.subheader("🗣️ Audience Reaction & Social Listening Intelligence")

        if not st.session_state.comments_analysis:
            st.info("No comments data available. Ensure 'Fetch & Analyze Audience Comments' is enabled in the sidebar.")
        else:
            ca = st.session_state.comments_analysis

            # Top Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Positive Sentiment</div>
                    <div class='metric-val' style='color:#15803d;'>{ca.get('positive_pct', 0)}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Discussion & Neutral</div>
                    <div class='metric-val' style='color:#b45309;'>{ca.get('neutral_pct', 0)}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Critical / Skeptical</div>
                    <div class='metric-val' style='color:#dc2626;'>{ca.get('critical_pct', 0)}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Audience Sample</div>
                    <div class='metric-val' style='color:#1d4ed8;'>{len(st.session_state.comments)} Top Comments</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Vibe Tags
            vibe_badges = "".join([f"<span class='badge-pill badge-purple'>✨ {tag}</span>" for tag in ca.get("vibe_tags", [])])
            st.markdown(f"**Audience Vibe:** {vibe_badges}", unsafe_allow_html=True)

            # Community Summary Card
            st.markdown(f"""
            <div style='background: rgba(59, 130, 246, 0.12); border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; margin: 12px 0 20px 0;'>
                <strong>💬 Community Reception:</strong> {ca.get('community_summary', '')}
            </div>
            """, unsafe_allow_html=True)

            # Two Column Split: What Viewers Loved vs Debates
            col_loved, col_debates = st.columns(2)
            with col_loved:
                st.markdown("#### ❤️ What Resonated Most with Viewers")
                for item in ca.get("what_viewers_loved", []):
                    st.markdown(f"- {item}")

            with col_debates:
                st.markdown("#### ⚖️ Community Debates & Counter-Arguments")
                for item in ca.get("debates_and_skepticism", []):
                    st.markdown(f"- {item}")

            st.markdown("---")

            # Questions & Topic Requests
            col_q, col_future = st.columns(2)
            with col_q:
                st.markdown("#### ❓ Top Questions Asked by Viewers")
                for q in ca.get("top_viewer_questions", []):
                    st.markdown(f"- ❓ *\"{q}\"*")

            with col_future:
                st.markdown("#### 🔮 What Viewers Want in Next Video")
                for topic in ca.get("future_topic_requests", []):
                    st.markdown(f"- 💡 {topic}")

            st.markdown("---")

            # Searchable Comments Feed with Timestamps
            st.markdown("#### 💬 Top Comments Feed")
            comment_filter = st.text_input("🔍 Filter comments by keyword:", "", key="cmt_filter")

            filtered_cmts = st.session_state.comments
            if comment_filter:
                filtered_cmts = [c for c in filtered_cmts if comment_filter.lower() in c.get("text", "").lower()]

            st.caption(f"Showing {len(filtered_cmts)} comments:")
            feed_container = st.container(height=450)
            with feed_container:
                for c in filtered_cmts:
                    author = c.get("author", "User")
                    likes = c.get("likes", "0")
                    published = c.get("published", "")
                    body = c.get("text", "")
                    ts_list = c.get("timestamps", [])

                    ts_html = ""
                    for t in ts_list:
                        ts_html += f"<a class='ts-tag' href='{t['url']}' target='_blank'>⏱️ {t['timestamp']}</a>"

                    st.markdown(f"""
                    <div class='comment-card'>
                        <div class='comment-author'>{author} <span style='font-weight:normal; color:#64748b;'>• {published}</span></div>
                        <div class='comment-meta'>👍 {likes} likes {ts_html}</div>
                        <div class='comment-body'>{body}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================
    # TAB 3: EXECUTIVE SUMMARY
    # ==========================================
    with tab_summary:
        st.subheader("📋 Executive Video Analysis & Summary")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            if st.button("🔄 Regenerate Executive Summary"):
                with st.spinner("Synthesizing transcript with AI..."):
                    st.session_state.summary = st.session_state.rag_engine.generate_summary(
                        st.session_state.full_transcript
                    )
                    st.rerun()
        else:
            st.write("Generate a comprehensive executive summary covering core insights, major breakthroughs, and action items.")
            if st.button("⚡ Generate Executive Summary Now", type="primary"):
                with st.spinner("Synthesizing transcript with AI..."):
                    st.session_state.summary = st.session_state.rag_engine.generate_summary(
                        st.session_state.full_transcript
                    )
                    st.rerun()

    # ==========================================
    # TAB 4: FULL TRANSCRIPT
    # ==========================================
    with tab_transcript:
        st.subheader("📜 Searchable Video Transcript")
        search_query = st.text_input("🔍 Search transcript for keywords:", "", key="tscript_filter")
        
        filtered_snippets = st.session_state.snippets or []
        if search_query:
            filtered_snippets = [
                s for s in filtered_snippets 
                if search_query.lower() in s.get("text", "").lower()
            ]
            st.caption(f"Found {len(filtered_snippets)} matching dialogue lines.")

        st.markdown(f"**Total Dialogue Cues:** {len(filtered_snippets)}")
        transcript_container = st.container(height=500)
        with transcript_container:
            for snip in filtered_snippets:
                sec = snip.get("start", 0.0)
                ts_str = format_timestamp(sec)
                url = f"https://youtu.be/{st.session_state.video_id}?t={int(sec)}"
                txt = snip.get("text", "")
                st.markdown(f"**[`{ts_str}`]({url})** {txt}")
