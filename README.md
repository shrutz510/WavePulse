# WavePulse
WavePulse: Real-time Content Analytics of Radio Livestreams

Radio remains a pervasive medium for mass information dissemination, with AM/FM stations reaching more Americans than either smartphone-based social networking or live television. Increasingly, radio broadcasts are also streamed online and accessed over the Internet. We present WavePulse, a framework that records, documents, and analyzes radio content in real-time. While our framework is generally applicable, we showcase the efficacy of WavePulse in a collaborative project with a team of political scientists focusing on the 2024 Presidential Elections. We use WavePulse to monitor livestreams of 396 news radio stations over a period of three months, processing close to 500,000 hours of audio streams. These streams were converted into time-stamped, diarized transcripts and analyzed to track answer key political science questions at both the national and state levels. Our analysis revealed how local issues interacted with national trends, providing insights into information flow. Our results demonstrate WavePulse's efficacy in capturing and analyzing content from radio livestreams sourced from the Web.

[Paper](https://arxiv.org/abs/2412.17998) | [Website](https://wave-pulse.io) | [Dataset - Raw](https://huggingface.co/datasets/nyu-dice-lab/wavepulse-radio-raw-transcripts) | [Dataset - Summarized](
https://huggingface.co/datasets/nyu-dice-lab/wavepulse-radio-summarized-transcripts)


![figure](/assets/overview.png)

## Overview
WavePulse is an end-to-end framework for recording, transcribing, and analyzing radio livestreams in real-time. It processes multiple concurrent audio streams into timestamped, speaker-diarized transcripts and enables content analysis through state-of-the-art AI models.

## Features
- Record multiple concurrent radio streams
- Convert speech to text with advanced speaker diarization
- Classify content (political vs non-political, ads vs content)
- Detect emerging narratives and track their spread
- Analyze sentiment and opinion trends
- Interactive visualization dashboard

## Requirements

### Minimum System Requirements
- Linux-based OS (tested on Ubuntu 20.04+)
- Python 3.8 or higher
- CUDA-enabled GPU (1 per 50 streams)
- At least 16GB RAM
- Storage space proportional to number of streams (2GB/stream - only transcription)

### API Access
- Hugging Face account with API token
- Google AI Studio account with API key

### Software Dependencies  
- FFmpeg
- CUDA toolkit (if using GPU)
- Other Python packages (specified in requirements.yml)

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/mittalgovind/wavepulse.git
cd wavepulse

# Set up conda environment
conda env create -f requirements.yml
conda activate wavepulse

# Install ffmpeg
sudo apt update && sudo apt install ffmpeg
```

### 2. API Setup
1. Get a Hugging Face token at https://huggingface.co/settings/tokens
2. Accept model agreements:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Get a Google AI Studio API key at https://ai.google.dev/aistudio

### 3. Configure Radio Sources
Edit `assets/weekly_schedule.json`:
```json
[
  {
    "url": "https://example.com/stream",
    "radio_name": "WXYZ",
    "time": [
      ["08:00", "14:00"], 
      ["17:00", "22:00"]
    ],
    "state": "NY"
  }
]
```

### 4. Start WavePulse

Full pipeline with all features:
```bash
python src/wavepulse.py --diarize \
  --hf-token YOUR_HUGGINGFACE_TOKEN \
  --gemini-api-key YOUR_GEMINI_KEY
```

Without content classification:
```bash
python src/wavepulse.py --diarize \
  --hf-token YOUR_HUGGINGFACE_TOKEN \
  --stop-classification
```

Basic recording and transcription only:
```bash
python src/wavepulse.py --stop-classification
```
All Parameters : [ARGS.md](ARGS.md)

## Component Control
Control individual components with these flags:
- `--stop-recording`: Disable audio recording
- `--stop-transcription`: Disable transcription 
- `--stop-classification`: Disable content classification
- `--diarize`: Enable speaker diarization (requires HF token)

## File Structure
```
assets/
├── data/
│   ├── recordings/              # Raw audio files
│   ├── audio_buffer_*/          # Processing buffers
│   └── transcripts/
│       ├── unclassified_buffer/ # Raw transcripts
│       └── classified/          # Processed transcripts
└── analytics/
    ├── transcripts/            # Analysis input
    └── sentiment_analysis/     # Analysis results
```

## Analytics Pipeline

### Run Sentiment Analysis
```bash
# First copy transcripts to analytics input folder as a clean copy
cp -r assets/data/transcripts/classified/* assets/analytics/transcripts/

# Run analysis
python src/analytics/sentiment/sentiment_analysis.py
python src/analytics/sentiment/calculate_metrics.py

# Results appear in assets/analytics/sentiment_analysis/
```

### Track Narratives

Summarize your transcripts
```
cd src/analytics/track_narratives
export GCP_API_KEY=YOUR_GCP_KEY
python ../summarizer.py \
      -i /path/to/raw/transcripts \
      -o /output/path
```

Create index (Vector) database and merge if you have multiple indices
```
python embed_summaries.py -i /path/to/summarized_text_files --batch_size 10
python merge_embeddings.py -i . -o merged_temp.h5
```

Talk to your database using top 5 most relevant retrieved summaries. 
```
python run_rag.py --k 5
```

For the election specific claim in the first case study of paper, see
[this file](src/analytics/track_narratives/election_specific_match.py)

## Process Flow
1. Audio recorder captures streams in 30-minute segments
2. Transcriber processes audio files through ASR pipeline
3. Content classifier categorizes transcript segments
4. Analytics pipeline processes classified transcripts
5. Results viewable through analytics dashboard

## Troubleshooting

### Common Issues
- **Stream connection errors**: Check URL validity and network connection
- **GPU memory errors**: Reduce concurrent transcription threads
- **Missing transcripts**: Verify ffmpeg installation and audio format
- **Classification delays**: Check Google API quotas and key validity

### Getting Help
- Check logs in `logs/` directory
- Open an issue on GitHub
- See documentation in `docs/`


## License
This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation
If you use WavePulse or any of its components, please cite:
```bibtex
@article{mittal2024wavepulse,
    title={WavePulse: Real-time Content Analytics of Radio Livestreams},
    author={Mittal, Govind and Gupta, Sarthak and Wagle, Shruti and Chopra, Chirag and DeMattee, Anthony J and Memon, Nasir and Ahamad, Mustaque and Hegde, Chinmay},
    journal={arXiv preprint arXiv:2412.17998},
    year={2024},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    eprint={2412.17998}
}
```
