"""
Call Analysis Dashboard - Streamlit App
Real-time transcription and sentiment analysis for customer service calls
"""

import streamlit as st
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from cache_manager import TranscriptionCache
from transcribe import ResultAccumulator, SentimentAnalyzer
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Call Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for color coding
st.markdown("""
<style>
.positive-sentiment {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
    color: #155724;
}
.negative-sentiment {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
    color: #721c24;
}
.neutral-sentiment {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
    color: #856404;
}
.transcript-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
    max-height: 400px;
    overflow-y: auto;
}
.partial-text {
    color: #6c757d;
    font-style: italic;
}
.final-text {
    color: #212529;
    font-weight: 500;
    margin: 5px 0;
}
.cache-badge {
    background-color: #17a2b8;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.8em;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcription_running' not in st.session_state:
    st.session_state.transcription_running = False
if 'transcript_segments' not in st.session_state:
    st.session_state.transcript_segments = []
if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'cache_hit' not in st.session_state:
    st.session_state.cache_hit = False

# Initialize cache
cache = TranscriptionCache()


def extract_sentiment_color(analysis_text: str) -> str:
    """Extract the sentiment keyword from the beginning of the analysis."""
    first_line = analysis_text.strip().split('\n')[0].upper()
    if 'POSITIVE' in first_line:
        return 'positive'
    elif 'NEGATIVE' in first_line:
        return 'negative'
    elif 'NEUTRAL' in first_line:
        return 'neutral'
    # Try to find it in the first 200 characters
    first_part = analysis_text[:200].upper()
    if 'POSITIVE' in first_part:
        return 'positive'
    elif 'NEGATIVE' in first_part:
        return 'negative'
    else:
        return 'neutral'


def process_audio_file(audio_file, language="en-US"):
    """Process uploaded audio file with transcription - simplified version."""
    # Get Azure credentials
    speech_key = os.getenv('AZURE_REAL_TIME_KEY')
    speech_endpoint = os.getenv('AZURE_REAL_TIME_ENDPOINT')

    if not speech_key or not speech_endpoint:
        st.error("Azure Speech credentials not found in .env file")
        return None

    # Reset file pointer to beginning
    audio_file.seek(0)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as tmp_file:
        content = audio_file.read()
        tmp_file.write(content)
        tmp_file.flush()
        tmp_path = tmp_file.name

    # Verify file was written
    file_size = os.path.getsize(tmp_path)
    if file_size == 0:
        st.error(f"Error: Audio file is empty (0 bytes)")
        os.unlink(tmp_path)
        return None

    st.info(f"Processing {file_size / 1024 / 1024:.2f} MB audio file...")

    try:
        # Configure Speech SDK
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=speech_endpoint)
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # Result accumulator
        accumulator = ResultAccumulator(input_source="file", audio_file=audio_file.name, language=language)

        # Storage for results (thread-safe)
        all_results = []
        done = [False]  # Use list to make it mutable in nested function
        error_msg = [None]

        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text
                offset = evt.result.offset // 10000
                duration = evt.result.duration // 10000
                all_results.append({'text': text, 'offset': offset, 'duration': duration})
                accumulator.add_final_result(text, offset, duration)

        def canceled_cb(evt):
            cancellation_details = speechsdk.CancellationDetails.from_result(evt.result)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                error_msg[0] = cancellation_details.error_details
            done[0] = True

        def session_stopped_cb(evt):
            done[0] = True

        # Connect callbacks
        recognizer.recognized.connect(recognized_cb)
        recognizer.canceled.connect(canceled_cb)
        recognizer.session_stopped.connect(session_stopped_cb)

        # Start recognition
        recognizer.start_continuous_recognition()

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Wait for completion
        start_time = time.time()
        while not done[0]:
            elapsed = time.time() - start_time
            if elapsed > 300:  # 5 minute timeout
                break

            # Update progress (fake progress based on time)
            progress = min(0.9, elapsed / 10.0)  # Ramp up to 90% over 10 seconds
            progress_bar.progress(progress)
            status_text.text(f"Processing... {len(all_results)} segments found")
            time.sleep(0.5)

        # Stop recognition
        recognizer.stop_continuous_recognition()
        time.sleep(0.5)

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # Check for errors
        if error_msg[0]:
            st.error(f"Recognition error: {error_msg[0]}")
            os.unlink(tmp_path)
            return None

        # Finalize
        accumulator.finalize()

        # Display results
        if all_results:
            st.success(f"Found {len(all_results)} speech segments")

            # Show transcript
            transcript_text = "\n\n".join([r['text'] for r in all_results])
            st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)

        # Clean up temp file
        os.unlink(tmp_path)

        return accumulator

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        import traceback
        st.code(traceback.format_exc())
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


def perform_analysis(transcript: str):
    """Perform sentiment analysis on the transcript."""
    # Check cache first
    cached_result = cache.check_duplicate(transcript)

    if cached_result:
        st.session_state.cache_hit = True
        return cached_result.get('sentiment_analysis', {}).get('analysis', '')

    st.session_state.cache_hit = False

    # Get OpenAI credentials
    openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    if not all([openai_endpoint, openai_api_key, openai_deployment]):
        st.error("Azure OpenAI credentials not found in .env file")
        return None

    # Create analyzer
    analyzer = SentimentAnalyzer(
        endpoint=openai_endpoint,
        api_key=openai_api_key,
        deployment=openai_deployment
    )

    # Perform analysis (without streaming for simplicity in dashboard)
    result = analyzer.analyze(transcript, stream_display=False)

    # Cache the result
    cache_data = {
        'full_transcript': transcript,
        'sentiment_analysis': result,
        'timestamp': datetime.now().isoformat()
    }
    cache.save_transcription(transcript, cache_data)

    return result.get('analysis', '')


# Dashboard Layout
st.title("Call Analysis Dashboard")
st.markdown("Real-time transcription and sentiment analysis for customer service calls")

# Sidebar
with st.sidebar:
    st.header("Settings")

    language = st.selectbox(
        "Recognition Language",
        ["en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR"],
        index=0
    )

    max_tokens = st.slider(
        "Analysis Max Tokens",
        min_value=500,
        max_value=4000,
        value=2000,
        step=500,
        help="Maximum length of the analysis"
    )

    st.markdown("---")
    st.header("Cache Stats")
    cache_stats = cache.get_cache_stats()
    st.metric("Cached Calls", cache_stats['total_cached'])
    st.metric("Cache Size", f"{cache_stats['cache_size_mb']:.2f} MB")

    if st.button("Clear Cache"):
        cache.clear_cache()
        st.success("Cache cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown("### Sentiment Colors")
    st.markdown('<div class="positive-sentiment">Positive</div>', unsafe_allow_html=True)
    st.markdown('<div class="neutral-sentiment">Neutral</div>', unsafe_allow_html=True)
    st.markdown('<div class="negative-sentiment">Negative</div>', unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["Upload File", "Live Recording", "History"])

with tab1:
    st.header("Upload Audio File")
    st.markdown("Upload a WAV or MP3 file for transcription and analysis")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Best results with WAV files (16kHz, mono)"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

        col1, col2 = st.columns([1, 4])
        with col1:
            process_button = st.button("Process Call", type="primary", use_container_width=True)

        if process_button:
            st.session_state.transcript_segments = []
            st.session_state.analysis_result = None
            st.session_state.cache_hit = False

            with st.spinner("🎙️ Transcribing audio..."):
                st.markdown("### Real-time Transcription")
                accumulator = process_audio_file(uploaded_file, language)

                if accumulator:
                    st.session_state.full_transcript = accumulator.get_full_transcript()
                    st.success(f"Transcription complete! ({len(st.session_state.full_transcript)} characters)")

                    # Show statistics
                    stats = accumulator.get_statistics()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Segments", stats['final_segments'])
                    with col2:
                        st.metric("Duration", f"{stats['total_duration_ms'] / 1000:.1f}s")
                    with col3:
                        st.metric("Partial Updates", stats['partial_segments'])

            if st.session_state.full_transcript:
                with st.spinner("🤖 Analyzing call..."):
                    st.markdown("### Sentiment Analysis")
                    analysis = perform_analysis(st.session_state.full_transcript)

                    if analysis:
                        st.session_state.analysis_result = analysis

                        # Show cache status
                        if st.session_state.cache_hit:
                            st.success("Retrieved from cache (saved API call!)")

                        # Determine sentiment color
                        sentiment_color = extract_sentiment_color(analysis)

                        # Display with color coding
                        st.markdown(
                            f'<div class="{sentiment_color}-sentiment">{analysis}</div>',
                            unsafe_allow_html=True
                        )

with tab2:
    st.header("Live Microphone Recording")
    st.info("Microphone recording coming soon! Use file upload for now.")
    st.markdown("""
    This feature will allow you to:
    - Record directly from your microphone
    - See real-time transcription as you speak
    - Analyze the call immediately after ending the recording
    """)

with tab3:
    st.header("Call History")
    st.markdown("Previously analyzed calls from cache")

    cached_items = cache.list_cached_items()

    if cached_items:
        for item in cached_items:
            with st.expander(f"{item['timestamp'][:19]} - {item['preview']}"):
                st.text(f"Cached: {item['timestamp']}")
                st.text(f"Hash: {item['hash']}")
    else:
        st.info("No cached calls yet. Analyze a call to see it here!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d;'>
    Powered by Azure Speech Services & Azure OpenAI | DeepSeek-V3.2
    </div>
    """,
    unsafe_allow_html=True
)
