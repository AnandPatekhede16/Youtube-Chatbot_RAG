import html
import re
from typing import Optional

def strip_markdown_for_speech(text: str) -> str:
    """Removes URLs, markdown symbols, and brackets so Text-to-Speech sounds natural."""
    # Remove URLs
    cleaned = re.sub(r'https?://\S+', '', text)
    # Remove markdown link text brackets like [14:20]
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
    # Remove code blocks
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
    # Remove markdown characters (*, _, #, `, ~, >)
    cleaned = re.sub(r'[\*\_#`~>]', '', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def render_sync_player_html(video_id: str, start_seconds: int = 0, autoplay: bool = False, height: int = 340) -> str:
    """
    Renders an embedded, responsive YouTube player that can start at a specific timestamp.
    """
    autoplay_param = "1" if autoplay else "0"
    embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}?start={int(start_seconds)}&autoplay={autoplay_param}&rel=0&modestbranding=1"

    return f"""
    <div style="width: 100%; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.18); background: #000;">
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
            <iframe 
                id="tube-mind-player"
                src="{embed_url}" 
                title="TubeMind Video Player"
                style="position: absolute; top:0; left:0; width:100%; height:100%; border:0;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """

def render_voice_input_html() -> str:
    """
    Returns a responsive Web Speech API voice dictation widget.
    Allows users to dictate their query via microphone with one-click copy.
    """
    return """
    <div id="voice-widget-container" style="
        background: rgba(125, 125, 125, 0.08);
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <button id="mic-btn" onclick="toggleRecognition()" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: #ff1e56;
                    color: #ffffff;
                    border: none;
                    border-radius: 20px;
                    padding: 6px 14px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    box-shadow: 0 2px 8px rgba(255, 30, 86, 0.35);
                ">
                    <span id="mic-icon">🎙️</span>
                    <span id="mic-text">Ask with Voice</span>
                </button>
                <span id="status-indicator" style="font-size: 0.82rem; opacity: 0.75;">Click mic to speak</span>
            </div>
            <button id="copy-btn" onclick="copyTranscript()" style="
                display: none;
                background: rgba(125, 125, 125, 0.15);
                color: inherit;
                border: 1px solid rgba(125, 125, 125, 0.3);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 0.80rem;
                font-weight: 500;
                cursor: pointer;
            ">📋 Copy to Chat</button>
        </div>
        <div id="transcript-result" style="
            display: none;
            margin-top: 8px;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.04);
            border-radius: 6px;
            font-size: 0.88rem;
            color: inherit;
            line-height: 1.4;
            border-left: 3px solid #ff1e56;
        "></div>
    </div>

    <script>
        let recognition = null;
        let isListening = false;
        let finalTranscript = '';

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isListening = true;
                document.getElementById('mic-btn').style.background = '#dc2626';
                document.getElementById('mic-icon').innerText = '🔴';
                document.getElementById('mic-text').innerText = 'Listening...';
                document.getElementById('status-indicator').innerText = 'Listening to your microphone...';
            };

            recognition.onresult = function(event) {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                const display = finalTranscript || interim;
                if (display) {
                    const resultBox = document.getElementById('transcript-result');
                    resultBox.style.display = 'block';
                    resultBox.innerText = '"' + display + '"';
                    document.getElementById('copy-btn').style.display = 'inline-block';
                }
            };

            recognition.onerror = function(event) {
                isListening = false;
                resetMicUI();
                document.getElementById('status-indicator').innerText = 'Voice error: ' + (event.error || 'Check mic permission');
            };

            recognition.onend = function() {
                isListening = false;
                resetMicUI();
                document.getElementById('status-indicator').innerText = 'Speech recognized! Click Copy to Chat.';
            };
        } else {
            document.getElementById('status-indicator').innerText = 'Web Speech not supported in this browser. Use Chrome, Edge, or Safari.';
        }

        function toggleRecognition() {
            if (!recognition) return;
            if (isListening) {
                recognition.stop();
            } else {
                finalTranscript = '';
                try {
                    recognition.start();
                } catch (e) {
                    recognition.stop();
                }
            }
        }

        function resetMicUI() {
            document.getElementById('mic-btn').style.background = '#ff1e56';
            document.getElementById('mic-icon').innerText = '🎙️';
            document.getElementById('mic-text').innerText = 'Ask with Voice';
        }

        function copyTranscript() {
            if (!finalTranscript) return;
            navigator.clipboard.writeText(finalTranscript).then(function() {
                const btn = document.getElementById('copy-btn');
                btn.innerText = '✅ Copied!';
                setTimeout(() => { btn.innerText = '📋 Copy to Chat'; }, 2000);
            });
        }
    </script>
    """

def render_tts_player_html(text_to_speak: str, button_id: str) -> str:
    """
    Renders an inline Text-to-Speech audio player button using window.speechSynthesis.
    """
    clean_text = strip_markdown_for_speech(text_to_speak)
    escaped_text = html.escape(clean_text).replace('"', '&quot;').replace("'", "&#39;")

    return f"""
    <div style="display: inline-block; margin-top: 6px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <button id="tts-btn-{button_id}" onclick="playTTS_{button_id}()" style="
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(125, 125, 125, 0.12);
            color: inherit;
            border: 1px solid rgba(125, 125, 125, 0.25);
            border-radius: 6px;
            padding: 3px 9px;
            font-size: 0.78rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        ">
            <span id="tts-icon-{button_id}">🔊</span>
            <span id="tts-label-{button_id}">Listen to Answer</span>
        </button>
    </div>

    <script>
        function playTTS_{button_id}() {{
            const btn = document.getElementById('tts-btn-{button_id}');
            const label = document.getElementById('tts-label-{button_id}');
            const icon = document.getElementById('tts-icon-{button_id}');

            if (window.speechSynthesis.speaking) {{
                window.speechSynthesis.cancel();
                label.innerText = 'Listen to Answer';
                icon.innerText = '🔊';
                btn.style.borderColor = 'rgba(125, 125, 125, 0.25)';
                return;
            }}

            const text = "{escaped_text}";
            if (!text) return;

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;

            utterance.onstart = function() {{
                label.innerText = 'Stop Audio';
                icon.innerText = '⏹️';
                btn.style.borderColor = '#ff1e56';
                btn.style.color = '#ff1e56';
            }};

            utterance.onend = function() {{
                label.innerText = 'Listen to Answer';
                icon.innerText = '🔊';
                btn.style.borderColor = 'rgba(125, 125, 125, 0.25)';
                btn.style.color = 'inherit';
            }};

            utterance.onerror = function() {{
                label.innerText = 'Listen to Answer';
                icon.innerText = '🔊';
                btn.style.borderColor = 'rgba(125, 125, 125, 0.25)';
                btn.style.color = 'inherit';
            }};

            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
