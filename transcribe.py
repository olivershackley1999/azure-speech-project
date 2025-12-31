#!/usr/bin/env python3
"""
Azure Speech-to-Text Streaming Transcription Script

Supports both microphone and file input with real-time JSON streaming display.
"""

import os
import sys
import json
import time
import argparse
import uuid
from datetime import datetime
from typing import Optional, Dict, List
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from openai import OpenAI


class ResultAccumulator:
    """Collects and structures all transcription results for final JSON output."""

    def __init__(self, input_source: str, audio_file: Optional[str] = None, language: str = "en-US"):
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.end_time = None
        self.input_source = input_source
        self.audio_file = audio_file
        self.language = language
        self.segments: List[Dict] = []
        self.final_texts: List[str] = []
        self.sentiment_analysis: Optional[Dict] = None

    def add_partial_result(self, text: str, offset: int, duration: int):
        """Store partial recognition result."""
        if text.strip():  # Only store non-empty results
            segment = {
                "type": "partial",
                "text": text,
                "offset_ms": offset,
                "duration_ms": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.segments.append(segment)

    def add_final_result(self, text: str, offset: int, duration: int):
        """Store final recognition result."""
        if text.strip():  # Only store non-empty results
            segment = {
                "type": "final",
                "text": text,
                "offset_ms": offset,
                "duration_ms": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.segments.append(segment)
            self.final_texts.append(text)

    def finalize(self):
        """Mark the session as complete."""
        self.end_time = datetime.now()

    def get_full_transcript(self) -> str:
        """Return concatenated final transcript."""
        return " ".join(self.final_texts)

    def set_sentiment_analysis(self, sentiment_data: Dict):
        """Store sentiment analysis results."""
        self.sentiment_analysis = sentiment_data

    def get_statistics(self) -> Dict:
        """Calculate statistics about the transcription."""
        final_segments = [s for s in self.segments if s["type"] == "final"]
        partial_segments = [s for s in self.segments if s["type"] == "partial"]

        total_duration = 0
        if final_segments:
            # Calculate total duration from final segments
            for segment in final_segments:
                total_duration = max(total_duration, segment["offset_ms"] + segment["duration_ms"])

        return {
            "total_segments": len(self.segments),
            "final_segments": len(final_segments),
            "partial_segments": len(partial_segments),
            "total_duration_ms": total_duration
        }

    def get_complete_json(self) -> Dict:
        """Return complete JSON structure."""
        result = {
            "metadata": {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "input_source": self.input_source,
                "audio_file": self.audio_file,
                "language": self.language
            },
            "segments": self.segments,
            "full_transcript": self.get_full_transcript(),
            "statistics": self.get_statistics()
        }

        # Add sentiment analysis if available
        if self.sentiment_analysis:
            result["sentiment_analysis"] = self.sentiment_analysis

        return result

    def save_to_file(self, filename: str):
        """Write JSON to file with pretty formatting."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.get_complete_json(), f, indent=2, ensure_ascii=False)
            f.write('\n')  # POSIX compliance


class RealTimeDisplayHandler:
    """Manages console output with visual formatting."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.last_partial_length = 0

    def display_partial(self, text: str):
        """Overwrite current line with partial text."""
        if self.quiet:
            return

        # Clear previous line and write new partial
        sys.stdout.write('\r' + ' ' * self.last_partial_length + '\r')
        display_text = f"[PARTIAL] {text}"
        sys.stdout.write(display_text)
        sys.stdout.flush()
        self.last_partial_length = len(display_text)

    def display_final(self, text: str):
        """Print final result on new line with formatting."""
        if self.quiet:
            return

        # Clear partial line first
        if self.last_partial_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_partial_length + '\r')
            self.last_partial_length = 0

        # Print final result
        sys.stdout.write('\n' + '=' * 60 + '\n')
        sys.stdout.write('[FINAL] ')
        sys.stdout.write(text)
        sys.stdout.write('\n' + '=' * 60 + '\n')
        sys.stdout.flush()

    def display_status(self, message: str):
        """Show status message on STDERR."""
        print(f"[STATUS] {message}", file=sys.stderr)


class SentimentAnalyzer:
    """Performs sentiment analysis on transcription using Azure OpenAI."""

    def __init__(self, endpoint: str, api_key: str, deployment: str):
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )
        self.deployment = deployment

    def analyze(self, transcript: str, stream_display: bool = True) -> Dict:
        """Perform detailed sentiment analysis on the transcript with real-time streaming."""
        if not transcript or not transcript.strip():
            return {
                "error": "No transcript available for analysis",
                "sentiment": None
            }

        system_prompt = """You are an expert customer service and sales call quality analyst. Your PRIMARY job is to critically assess the performance of the agent/representative on this call.

CRITICAL ASSESSMENT STANDARDS:
- Be STRICT in your evaluation. This is a professional quality assurance review.
- Even MILD rudeness, abrasiveness, impatience, or cutting off the customer should result in significant downmarks and penalties.
- Interrupting the customer, talking over them, or not allowing them to finish thoughts is a MAJOR negative.
- A rushed, dismissive, or condescending tone warrants harsh criticism.
- Only reward TRULY EXCEPTIONAL agents with positive overall sentiment. Average performance should be rated as NEUTRAL at best.
- The bar for "positive" is HIGH - the agent must demonstrate outstanding empathy, professionalism, active listening, and problem-solving.

Your analysis should be honest, critical, and focused on holding agents accountable to the highest customer service standards.

Write your analysis in clear, easy-to-read prose. Use headers and bullet points for organization, but DO NOT use JSON, code blocks, or technical formatting.

Analyze the following aspects:

CALL OVERVIEW
- Call type (sales, customer service, support, complaint resolution, etc.)
- Overall outcome (successful, unsuccessful, requires follow-up)
- Call duration estimate
- Talking time ratio: Estimate what percentage of the call the agent spoke vs. the customer

OVERALL SENTIMENT (for color coding)
Start with a single word: POSITIVE, NEGATIVE, or NEUTRAL

IMPORTANT:
- POSITIVE = Only for truly exceptional agent performance (rare - maybe 1 in 20 calls)
- NEUTRAL = Average or acceptable performance with no major issues
- NEGATIVE = Any significant issues, rudeness, poor listening, or below-average performance

Default to NEUTRAL or NEGATIVE. Be stingy with POSITIVE ratings.

SENTIMENT BREAKDOWN BY SPEAKER
- Customer Sentiment: (Positive, Negative, Neutral, Mixed) with confidence score
  - Initial sentiment when call started
  - Final sentiment when call ended
  - Key shifts and what triggered them
- Agent/Rep Sentiment: (Positive, Negative, Neutral, Mixed)
  - Tone and demeanor throughout
  - Consistency and professionalism
  - Emotional intelligence displayed

SENTIMENT PROGRESSION
- How did the customer's sentiment change from start to end?
- Critical moments: Identify any significant sentiment shifts and what triggered them
- Did the agent successfully improve or maintain positive sentiment?

CONVERSATION DYNAMICS
- Who spoke more: agent or customer?
- Talking time ratio (approximate percentage)
- Was the balance appropriate for the call type?
- Did the agent listen effectively or dominate the conversation?

INTENT IDENTIFICATION
- Primary customer intent (what did the customer want to achieve?)
- Secondary intents or concerns raised during the call
- Was the customer's intent successfully addressed?

AGENT PERFORMANCE - STRENGTHS (if any)
- What did the agent do exceptionally well? (Be selective - only note truly outstanding behaviors)
- Effective techniques used that went above and beyond basic expectations
- Positive communication patterns that demonstrate mastery
- Specific examples from the transcript

AGENT PERFORMANCE - CRITICAL ISSUES & AREAS FOR IMPROVEMENT
- Any instances of rudeness, impatience, abrasiveness, or condescension (even mild)
- Did the agent interrupt, talk over, or cut off the customer? (MAJOR NEGATIVE)
- Rushed responses or dismissive language
- Lack of empathy or active listening
- Missed opportunities to build rapport or demonstrate care
- Any tone issues or unprofessional language
- Communication gaps, unclear explanations, or confusing statements
- Specific examples from the transcript demonstrating poor performance
- Training deficiencies identified

CALL QUALITY SCORES (out of 10) - BE STRICT
Score honestly and critically. Average performance = 5-6. Exceptional = 9-10. Poor = 1-4.
Do NOT inflate scores. Be harsh where warranted.

- Professionalism level (deduct points for ANY unprofessional behavior)
- Empathy and active listening (deduct heavily for interruptions or dismissiveness)
- Problem resolution effectiveness
- Communication clarity (deduct for rushed, unclear, or confusing communication)
- Overall call quality score (reflect your critical assessment - rarely above 7)

KEY TOPICS & PAIN POINTS
- Main topics discussed
- Product/service mentioned
- Pain points or objections raised by customer

ACTIONABLE RECOMMENDATIONS
- Specific recommendations for the agent to improve future calls
- Quick wins: immediate actions the agent can take

IMPORTANT REMINDERS:
1. Start your response with one of these words for color coding: POSITIVE, NEGATIVE, or NEUTRAL
2. Your PRIMARY focus is critically evaluating the AGENT'S performance
3. Be STRICT and HONEST - do not sugarcoat or inflate scores
4. Call out ANY unprofessional behavior, even if subtle
5. POSITIVE ratings should be RARE - reserve for truly exceptional agents only
6. Use specific quotes from the transcript to support your critical analysis
7. Remember: This is quality assurance, not a participation trophy

Write in clear, natural language with a professional, critical tone."""

        try:
            # Use streaming API
            stream = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Please perform a detailed sentiment analysis on the following transcription:\n\n{transcript}"
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            # Collect the streamed response
            response_text = ""

            if stream_display:
                print("\n" + "="*70)
                print("CALL ANALYSIS")
                print("="*70 + "\n")

            for chunk in stream:
                # Check if chunk has choices and content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        response_text += content

                        if stream_display:
                            # Stream the text directly to screen
                            sys.stdout.write(content)
                            sys.stdout.flush()

            if stream_display:
                print("\n")

            # Store the analysis as plain text
            sentiment_data = {
                "analysis": response_text,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": self.deployment
            }

            return sentiment_data

        except Exception as e:
            return {
                "error": f"Sentiment analysis failed: {str(e)}",
                "sentiment": None
            }

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract and parse JSON from the model response."""
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If not JSON, return the raw text
            return {
                "raw_analysis": response_text,
                "model": self.deployment
            }


class TranscriptionEngine:
    """Manages Azure Speech SDK lifecycle and event handling."""

    def __init__(
        self,
        input_mode: str,
        speech_key: str,
        accumulator: ResultAccumulator,
        display_handler: RealTimeDisplayHandler,
        audio_file_path: Optional[str] = None,
        language: str = "en-US",
        speech_endpoint: Optional[str] = None,
        speech_region: Optional[str] = None
    ):
        self.input_mode = input_mode
        self.audio_file_path = audio_file_path
        self.accumulator = accumulator
        self.display_handler = display_handler
        self.done = False
        self.error_occurred = False

        # Configure Speech SDK - prefer endpoint over region
        if speech_endpoint:
            self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=speech_endpoint)
        elif speech_region:
            self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        else:
            raise ValueError("Either speech_endpoint or speech_region must be provided")

        self.speech_config.speech_recognition_language = language

        # Configure Audio Input
        if input_mode == "microphone":
            self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        elif input_mode == "file":
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            self.audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")

        # Create recognizer
        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config
        )

        # Setup callbacks
        self.setup_callbacks()

    def setup_callbacks(self):
        """Attach event handlers to recognizer."""

        def recognizing_callback(evt):
            """Handle partial results."""
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                text = evt.result.text
                offset = evt.result.offset // 10000  # Convert to milliseconds
                duration = evt.result.duration // 10000

                # Display to screen
                self.display_handler.display_partial(text)

                # Accumulate for JSON
                self.accumulator.add_partial_result(text, offset, duration)

        def recognized_callback(evt):
            """Handle final results."""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text
                offset = evt.result.offset // 10000
                duration = evt.result.duration // 10000

                # Display to screen
                self.display_handler.display_final(text)

                # Accumulate for JSON
                self.accumulator.add_final_result(text, offset, duration)
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                self.display_handler.display_status("No speech recognized in this segment")

        def canceled_callback(evt):
            """Handle errors and cancellation."""
            cancellation_details = speechsdk.CancellationDetails.from_result(evt.result)

            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                self.display_handler.display_status(
                    f"ERROR: {cancellation_details.error_details}"
                )
                self.error_occurred = True
            else:
                self.display_handler.display_status(
                    f"Canceled: {cancellation_details.reason}"
                )

            self.done = True

        def session_stopped_callback(evt):
            """Handle session termination."""
            self.display_handler.display_status(f"Session stopped")
            self.done = True

        def session_started_callback(evt):
            """Handle session start."""
            self.display_handler.display_status(f"Session started: {evt.session_id}")

        # Connect callbacks
        self.recognizer.recognizing.connect(recognizing_callback)
        self.recognizer.recognized.connect(recognized_callback)
        self.recognizer.canceled.connect(canceled_callback)
        self.recognizer.session_stopped.connect(session_stopped_callback)
        self.recognizer.session_started.connect(session_started_callback)

    def start_transcription(self):
        """Begin continuous recognition."""
        self.display_handler.display_status("Starting transcription...")
        self.recognizer.start_continuous_recognition()

    def stop_transcription(self):
        """Clean shutdown with error handling."""
        if self.recognizer:
            try:
                self.display_handler.display_status("Stopping transcription...")
                self.recognizer.stop_continuous_recognition()
                time.sleep(0.5)  # Brief wait for cleanup
            except Exception as e:
                print(f"Warning during shutdown: {e}", file=sys.stderr)


def main():
    """Main execution function."""

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Azure Speech-to-Text transcription with real-time streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe from microphone
  python transcribe.py --mic

  # Transcribe from audio file
  python transcribe.py --file Arthur.wav

  # Transcribe with sentiment analysis
  python transcribe.py --file Arthur.wav --analyze

  # Specify output file
  python transcribe.py --file Arthur.wav --output results.json

  # Use specific language
  python transcribe.py --mic --language es-ES

  # Microphone with sentiment analysis
  python transcribe.py --mic --analyze --output my_analysis.json
"""
    )

    # Input source (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--mic', '--microphone',
        action='store_true',
        help='Use microphone for real-time transcription'
    )
    input_group.add_argument(
        '--file',
        type=str,
        metavar='PATH',
        help='Path to audio file for transcription'
    )

    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='transcription.json',
        help='Output JSON file path (default: transcription.json)'
    )

    parser.add_argument(
        '--language', '-l',
        type=str,
        default='en-US',
        help='Recognition language (default: en-US)'
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        default=None,
        help='Azure region (optional, defaults to endpoint from .env file)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress real-time display, only save JSON'
    )

    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Perform sentiment analysis on the transcription using Azure OpenAI'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    speech_key = os.getenv('AZURE_REAL_TIME_KEY')
    speech_endpoint = os.getenv('AZURE_REAL_TIME_ENDPOINT')

    if not speech_key:
        print("ERROR: AZURE_REAL_TIME_KEY not found in .env file", file=sys.stderr)
        sys.exit(1)

    if not speech_endpoint:
        print("ERROR: AZURE_REAL_TIME_ENDPOINT not found in .env file", file=sys.stderr)
        sys.exit(1)

    # Load OpenAI credentials if sentiment analysis is requested
    openai_endpoint = None
    openai_api_key = None
    openai_deployment = None

    if args.analyze:
        openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

        if not openai_endpoint or not openai_api_key or not openai_deployment:
            print("ERROR: Sentiment analysis requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in .env file", file=sys.stderr)
            sys.exit(1)

    # Determine input mode
    if args.mic:
        input_mode = "microphone"
        audio_file = None
        print("Starting microphone transcription. Press Ctrl+C to stop.\n")
    else:
        input_mode = "file"
        audio_file = args.file
        print(f"Starting file transcription: {audio_file}\n")

    # Initialize components
    accumulator = ResultAccumulator(
        input_source=input_mode,
        audio_file=audio_file,
        language=args.language
    )
    display_handler = RealTimeDisplayHandler(quiet=args.quiet)

    # Create transcription engine
    try:
        engine = TranscriptionEngine(
            input_mode=input_mode,
            audio_file_path=audio_file,
            language=args.language,
            speech_key=speech_key,
            speech_endpoint=speech_endpoint,
            speech_region=args.region if args.region else None,
            accumulator=accumulator,
            display_handler=display_handler
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize transcription engine: {e}", file=sys.stderr)
        sys.exit(1)

    # Start transcription with error handling
    try:
        engine.start_transcription()

        # Wait for completion
        if input_mode == "file":
            # File mode: wait until done
            while not engine.done:
                time.sleep(0.1)
        else:
            # Microphone mode: wait for Ctrl+C
            print("\nListening... Press Ctrl+C to stop.\n")
            while not engine.done:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping transcription...", file=sys.stderr)

    finally:
        # Stop recognition
        engine.stop_transcription()

        # Finalize accumulator
        accumulator.finalize()

        # Perform sentiment analysis if requested
        if args.analyze and accumulator.get_full_transcript():
            print(f"\n\n{'=' * 60}")
            print("PERFORMING SENTIMENT ANALYSIS")
            print(f"{'=' * 60}")
            print("Analyzing transcription with Azure OpenAI (DeepSeek-V3.2)...", file=sys.stderr)
            print(f"Max tokens: 1000 | Streaming: Real-time\n", file=sys.stderr)

            try:
                analyzer = SentimentAnalyzer(
                    endpoint=openai_endpoint,
                    api_key=openai_api_key,
                    deployment=openai_deployment
                )
                # Stream the analysis in real-time
                sentiment_results = analyzer.analyze(
                    accumulator.get_full_transcript(),
                    stream_display=True
                )
                accumulator.set_sentiment_analysis(sentiment_results)

                # Check for errors
                if "error" in sentiment_results:
                    print(f"\nSentiment Analysis Error: {sentiment_results['error']}\n")
                else:
                    print(f"\n{'=' * 60}")
                    print("SENTIMENT ANALYSIS COMPLETE")
                    print(f"{'=' * 60}\n")

            except Exception as e:
                print(f"ERROR: Failed to perform sentiment analysis: {e}", file=sys.stderr)

        # Display summary
        print(f"\n\n{'=' * 60}")
        print("TRANSCRIPTION COMPLETE")
        print(f"{'=' * 60}")

        full_transcript = accumulator.get_full_transcript()
        if full_transcript:
            print(f"Full transcript:\n{full_transcript}\n")
        else:
            print("No speech was recognized.\n")

        stats = accumulator.get_statistics()
        print(f"Statistics:")
        print(f"  Total segments: {stats['total_segments']}")
        print(f"  Final segments: {stats['final_segments']}")
        print(f"  Partial segments: {stats['partial_segments']}")
        print(f"  Duration: {stats['total_duration_ms'] / 1000:.2f}s")

        # Save to file
        output_file = args.output
        try:
            accumulator.save_to_file(output_file)
            print(f"\nJSON saved to: {output_file}")
        except Exception as e:
            print(f"\nERROR: Failed to save JSON: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
