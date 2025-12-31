# Azure Speech Project

Automated pipeline for call transcription and subsequent analysis via LLM.

## Demo

- [Terminal Demo](https://olivershackley.com/assets/videos/az-speech-terminal-demo.mov) - Command-line transcription with real-time streaming
- [Streamlit Demo](https://olivershackley.com/assets/videos/az-speech-streamlit-demo.mov) - Interactive dashboard for call analysis

## What it does

- Transcribes audio files or live microphone input using Azure Speech-to-Text
- Displays real-time partial and final transcription results with streaming output
- Performs sentiment analysis on completed transcripts using Azure OpenAI (DeepSeek-V3.2)
- Generates call quality assessments with agent performance scoring
- Caches analysis results to avoid redundant API calls

## How it works

**Four-step pipeline:**

1. **Input**: Accepts WAV/MP3 audio files or live microphone input
2. **Transcription**: Azure Speech Services processes audio with continuous recognition, streaming partial results in real-time
3. **Sentiment Analysis**: Complete transcript sent to Azure OpenAI for detailed call quality analysis including sentiment progression, agent performance scores, and actionable recommendations
4. **Output**: Results saved as structured JSON with full transcript, segment timing, and analysis

## Tech Stack

- Azure Speech Services (real-time speech-to-text)
- Azure OpenAI (DeepSeek-V3.2 for sentiment analysis)
- Python 3.13
- Streamlit (interactive dashboard)

## Installation

**Prerequisites**: Azure Speech Services and Azure OpenAI credentials required.

1. Clone the repository
   ```bash
   git clone https://github.com/olivershackley1999/azure-speech-project.git
   cd azure-speech-project
   ```

2. Install dependencies
   ```bash
   pip install azure-cognitiveservices-speech openai python-dotenv streamlit
   ```

3. Configure credentials in `.env`
   ```
   AZURE_REAL_TIME_KEY=your_speech_key
   AZURE_REAL_TIME_ENDPOINT=your_speech_endpoint
   AZURE_OPENAI_ENDPOINT=your_openai_endpoint
   AZURE_OPENAI_API_KEY=your_openai_key
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   ```

4. Run transcription
   ```bash
   # From audio file
   python transcribe.py --file audio.wav --analyze

   # From microphone
   python transcribe.py --mic --analyze
   ```

5. Or launch the dashboard
   ```bash
   streamlit run dashboard.py
   ```

## Sample Output

```json
{
  "metadata": {
    "session_id": "abc-123",
    "input_source": "file",
    "language": "en-US"
  },
  "full_transcript": "Hello, I'm calling about my account...",
  "statistics": {
    "final_segments": 12,
    "total_duration_ms": 45000
  },
  "sentiment_analysis": {
    "analysis": "NEUTRAL\n\nCALL OVERVIEW\n- Call type: Customer service inquiry..."
  }
}
```

## Project Motivation

Built to explore automated call center quality assurance - combining real-time transcription with LLM-powered analysis to evaluate agent performance, track customer sentiment, and generate actionable coaching insights.
