# WavePulse Parameters

## Directory Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--assets-dir` | `assets` | Base data directory path |
| `--data-dir` | `data` | Data directory path |
| `--recordings-dir` | `recordings` | Directory for audio recordings |
| `--models-dir` | `models` | ML models directory path |
| `--audio-dir` | `audio` | Audio files directory |
| `--transcripts-dir` | `transcripts` | Transcript files directory |
| `--audio-buffer-dir` | `audio_buffer` | Temporary audio files directory |
| `--logs-dir` | `logs` | Logs directory |

## Radio Stream Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--radio-schedule` | `weekly_schedule.json` | Radio streams schedule file |
| `--segment-duration` | `1800` | Recording segment duration (seconds) |
| `--retries` | `3` | Number of connection retries |
| `--wait-time` | `60` | Wait time between retries (seconds) |

## Transcription Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--whisperx-model` | `large-v3` | WhisperX model selection |
| `--whisperx-batch-size` | `16` | Batch size for transcription |
| `--whisperx-compute-type` | `float16` | Computation type |
| `--device` | `cuda`/`cpu` | Processing device |
| `--diarize` | `False` | Enable speaker diarization |
| `--hf-token` | `""` | Hugging Face API token |

## Classification Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gemini-api-key` | `api-key` | Google Gemini API key |
| `--concurrent-classification` | `1` | Number of concurrent classifications |

## Component Control
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stop-transcription` | `False` | Disable transcription |
| `--stop-recording` | `False` | Disable recording |
| `--stop-classification` | `False` | Disable classification |

## Application Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timezone` | `US/Eastern` | Application timezone |
| `--no-of-repetition` | `1` | Scheduler restart count |
| `--shutdown-time` | `03:00` | Daily shutdown time |
| `--restart-time` | `03:10` | Daily restart time |

## Backup Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backup-audio` | `False` | Enable audio backup |
| `--ftp-server` | `172.24.113.102` | FTP server address |
| `--ftp-port` | `21` | FTP port |
| `--ftp-username` | `marvin` | FTP username |
| `--ftp-password` | `42` | FTP password |
| `--ftp-remote-folder` | `daily_recordings` | Remote backup folder |

## Analytics Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no_of_jobs` | `1` | Parallel jobs for sentiment analysis |
| `--sentiment_start_date` | `2024_06_26` | Analysis start date |
| `--sentiment_end_date` | `""` | Analysis end date |

