#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University

# External libraries
import torch

# Standard libraries
from argparse import ArgumentParser
from datetime import datetime, timedelta


def get_args():
    parser = ArgumentParser(description="Radio Observatory")

    # General directory structure
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="assets",
        help="path to base data directory (default: data)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="path to data directory (default: data)",
    )

    parser.add_argument(
        "--recordings-dir",
        type=str,
        default="recordings",
        help="path to data directory (default: recordings)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="path to ml models directory (default: models)",
    )

    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="test_data",
        help="path to test data directory (default: test_data)",
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default="audio",
        help="path to audio files in data directory (default: audio)",
    )

    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="transcripts",
        help="path to transcript files in data directory (default: transcripts)",
    )

    parser.add_argument(
        "--audio-buffer-dir",
        type=str,
        default="audio_buffer",
        help="path to temporary audio files in data directory pending transcription (default: temp)",
    )

    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="path to temporary audio files in data directory pending transcription (default: temp)",
    )

    parser.add_argument(
        "--factcheck_dir",
        type=str,
        default="factcheck_json",
        help="path to fact check json files (default: factcheck_json)",
    )

    parser.add_argument(
        "--fact_today_dir",
        action="store_true",
        help="path to today's scraped files (default: False)",
    )

    # For radio streamer
    parser.add_argument(
        "--radio-schedule",
        type=str,
        default="weekly_schedule.json",
        help="file that contains schedule for recording radio streams",
    )

    parser.add_argument(
        "--segment-duration",
        type=int,
        default=1800,
        help="audio segment duration to record in seconds (default:1800) in one file",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of retries per segment to connect to radio stream in case of failure.",
    )

    parser.add_argument(
        "--wait-time",
        type=int,
        default=60,
        help="wait time between consecutive retries in seconds (default: 60)",
    )

    # Transcription model arguments
    parser.add_argument(
        "--whisperx-model",
        type=str,
        default="large-v3",
        help=(
            "whisper model to use for transcription (default: large-v2) "
            "possible values: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, "
            "large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en"
        ),
    )

    parser.add_argument(
        "--whisperx-batch-size",
        type=int,
        default=16,
        help="batch size for whisperx model for transcription larger batch size will lead to faster transcription but then more GPU memory is needed (default: 16)",
    )

    parser.add_argument(
        "--whisperx-compute-type",
        type=str,
        default="float16",
        help="whisper model to use for transcription (default: float16)",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="hugging face token to use for diarization model (default: " ")",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    parser.add_argument(
        "--diarize", action="store_true", help="Enable the diarization (default: False)"
    )

    # For calssification
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default="api-key",
        help="api key to be used for calling gemini api",
    )

    parser.add_argument(
        "--concurrent-classification",
        type=int,
        default=1,
        help="number of transcripts to be classified concurrently using gemini based on api rate limits (default: 1)",
    )

    # Toggle feature on/off
    parser.add_argument(
        "--stop-transcription",
        action="store_true",
        help="Disable transcription (default: False)",
    )

    parser.add_argument(
        "--stop-recording",
        action="store_true",
        help="Disable recording radio streams (default: False)",
    )

    parser.add_argument(
        "--stop-classification",
        action="store_true",
        help="Disable transcript content classification (default: False)",
    )

    # General application cofigurations
    parser.add_argument(
        "--timezone",
        type=str,
        default="US/Eastern",
        help="timezone for scheduling (default: US/Eastern)",
    )

    parser.add_argument(
        "--no-of-repetition",
        type=int,
        default=1,
        help="number of times to restart scheduler (default: 1)",
    )

    parser.add_argument(
        "--shutdown-time",
        type=str,
        default="03:00",
        help="time when application will shutdown (default: 3:00)",
    )

    parser.add_argument(
        "--restart-time",
        type=str,
        default="03:10",
        help="time when application will restart (default: 3:10)",
    )

    # For audio backup
    parser.add_argument(
        "--backup-audio",
        action="store_true",
        help="Enable the storing audio files in backup (default: False)",
    )

    parser.add_argument(
        "--ftp-server",
        type=str,
        default="172.24.113.102",
        help="ftp server (default: server)",
    )

    parser.add_argument(
        "--ftp-port",
        type=str,
        default="21",
        help="ftp port (default: server)",
    )

    parser.add_argument(
        "--ftp-username",
        type=str,
        default="marvin",
        help="ftp username (default: marvin)",
    )

    parser.add_argument(
        "--ftp-password",
        type=str,
        default="42",
        help="ftp password (default: 42)",
    )

    parser.add_argument(
        "--ftp-remote-folder",
        type=str,
        default="daily_recordings",
        help="ftp password (default: daily_recordings)",
    )

    # For fact checker
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.today() - timedelta(1)).strftime("%m-%d-%Y"),
        help="The start date for crawling (MM-DD-YYYY)",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=(datetime.today() - timedelta(1)).strftime("%m-%d-%Y"),
        help="The end date for crawling (MM-DD-YYYY)",
    )

    parser.add_argument(
        "--start_page", type=int, default=1, help="The start page for crawling"
    )

    parser.add_argument(
        "--end_page", type=int, default=5, help="The end page for crawling"
    )

    parser.add_argument("--title_keys", type=str, default="", help="Keywords in title")

    parser.add_argument("--tags", type=str, default="", help="Tags in article")

    parser.add_argument(
        "--spiders", type=str, nargs="*", help="List of spiders to run (default: all)"
    )

    parser.add_argument(
        "--political_keywords",
        type=str,
        nargs="*",
        default=[
            "politics",
            "election",
            "government",
            "policy",
            "politician",
            "senate",
            "congress",
            "president",
            "pelosi",
            "campaign",
            "vote",
            "democracy",
            "legislation",
            "parliament",
            "minister",
            "diplomacy",
            "administration",
            "law",
            "regulation",
            "governance",
            "political",
            "party",
            "national",
            "immigration",
            "health",
            "biden",
            "trump",
            "presidential",
            "debate",
            "liberal",
            "conservative",
            "republican",
            "democrat",
            "socialism",
            "capitalism",
            "justice",
            "equality",
            "freedom",
            "rights",
            "liberty",
            "economy",
            "tax",
            "budget",
            "foreign",
            "domestic",
            "trade",
            "war",
            "peace",
            "security",
            "defense",
            "environment",
            "climate",
            "education",
            "welfare",
            "healthcare",
            "infrastructure",
            "transportation",
            "energy",
            "labor",
            "employment",
            "pension",
            "retirement",
            "crime",
            "justice",
            "court",
            "judge",
            "attorney",
            "lobbyist",
            "reform",
            "referendum",
            "constitution",
            "bill",
            "ordinance",
            "executive",
            "judicial",
            "legislative",
            "senator",
            "congressman",
            "governor",
            "mayor",
            "council",
            "caucus",
            "primary",
            "ballot",
            "protest",
            "activism",
            "campaign finance",
            "super PAC",
            "lawmaker",
            "whistleblower",
            "impeachment",
            "scandal",
            "gerrymandering",
            "obama",
            "clinton",
            "sanders",
            "warren",
            "mcconnell",
            "schumer",
            "aoc",
            "ocasio-cortez",
            "harris",
            "pence",
            "cruz",
            "rubio",
            "graham",
            "romney",
            "mccarthy",
            "boebert",
            "greene",
            "newsom",
            "deblasio",
            "cuomo",
            "desantis",
            "youngkin",
            "whitmer",
            "abbott",
            "kemp",
            "stacey abrams",
            "yellen",
            "powell",
            "garland",
            "fauci",
            "gorsuch",
            "kavanaugh",
            "barrett",
            "sotomayor",
            "kagan",
            "roberts",
            "thomas",
            "alito",
            "bush",
            "cheney",
            "barr",
            "sessions",
            "mueller",
            "comey",
        ],
        help="List of political keywords to filter articles",
    )

    # For analytics
    parser.add_argument(
        "--no_of_jobs",
        type=int,
        default=1,
        help="number of parallel jobs for sentimental analysis",
    )

    parser.add_argument(
        "--sentiment_start_date",
        type=str,
        default="2024_06_26",
        help="Keywords in title",
    )

    parser.add_argument(
        "--sentiment_end_date", type=str, default="", help="Keywords in title"
    )

    args = parser.parse_args()

    return args
