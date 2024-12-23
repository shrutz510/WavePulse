#!/bin/python

import argparse
import json
import multiprocessing
import math
import os
import shutil
import subprocess
import sys
import time

from loguru import logger
from sentiment_keywords import keyword_dict
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def get_processed_files(sentiment_buffer):
    folder_list = [
        folder for folder in os.listdir(sentiment_buffer) if not folder.startswith(".")
    ]
    file_list = []
    for folder in folder_list:
        folder_path = os.path.join(sentiment_buffer, folder)
        files = [file for file in os.listdir(folder_path) if not file.startswith(".")]
        file_list.extend(files)
    print(f"total processed files: {len(file_list)}")
    return file_list


def get_transcript_files(transcripts):
    files = [file for file in os.listdir(transcripts) if not file.startswith(".")]
    print(f"total transcript files: {len(files)}")
    return files


def get_unprocessed_files(sentiment_buffer, transcripts):
    """
    Get all unprocessed transcripts by comparing files in "transcripts" and "sentiment_buffer"
    """
    processed_file_list = get_processed_files(sentiment_buffer)
    transcript_file_list = get_transcript_files(transcripts)
    processed_file_set = set(processed_file_list)
    transcript_file_set = set(transcript_file_list)
    unprocessed_file_set = transcript_file_set - processed_file_set
    print(f"total files pending sentiment analysis: {len(unprocessed_file_set)}")
    return list(unprocessed_file_set)


def chunk_list(lst, n):
    """
    Splits a list into n equal chunks.
    """
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def sentiment_analysis(
    file_name, source_dir, dest_dir, sentiment_pipeline, keyword_dict
):
    """
    Do sentiment analysis sentence by sentence, for sentences containing predefined keywords, and save the results in a json file.
    """
    keywords_lower = {
        key: [keyword.lower() for keyword in keywords]
        for key, keywords in keyword_dict.items()
    }
    local_path = os.path.join(source_dir, file_name)

    try:
        with open(local_path, "r") as f:
            transcript = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: The file {local_path} was not found.")
        return None
    except json.JSONDecodeError:
        invalid_json_dir = "invalid_json_dir"
        logger.error(
            f"Error: The file {local_path} is not a valid JSON file. Moving it to {invalid_json_dir}."
        )
        os.makedirs(invalid_json_dir, exist_ok=True)
        invalid_json_path = os.path.join(invalid_json_dir, file_name)
        shutil.move(local_path, invalid_json_path)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the file: {e}")
        return None
    name = file_name.split(".")[0]
    name_details = name.split("_")
    state = name_details[0]
    call_sign = name_details[1]
    year = name_details[2]
    month = name_details[3]
    day = name_details[4]
    hour = name_details[5]
    minute = name_details[6]
    result = {key: [] for key in keyword_dict}
    flags = {key: 0 for key in keyword_dict}  # Initialize flags for each category to 0
    for seg in transcript:
        text = seg["text"].lower()

        for category, keywords in keywords_lower.items():
            if any(keyword in text for keyword in keywords):
                flags[category] = 1
        if sum(flags.values()) == 1:
            hf_result = sentiment_pipeline(text)[0]
            for key, value in flags.items():
                if value == 1:
                    result[key].append(
                        {
                            "label": hf_result["label"],
                            "score": hf_result["score"],
                            "text": text,
                        }
                    )
                    flags[key] = 0
        elif sum(flags.values()) > 1:
            hf_result = sentiment_pipeline(text)[0]
            if flags["Biden"] == 1 and flags["Trump"] == 1:
                result["Biden-Trump"].append(
                    {
                        "label": hf_result["label"],
                        "score": hf_result["score"],
                        "text": text,
                    }
                )
            if flags["Harris"] == 1 and flags["Trump"] == 1:
                result["Harris-Trump"].append(
                    {
                        "label": hf_result["label"],
                        "score": hf_result["score"],
                        "text": text,
                    }
                )
            if flags["Democrats"] == 1 and flags["Republicans"] == 1:
                result["Democrats-Republicans"].append(
                    {
                        "label": hf_result["label"],
                        "score": hf_result["score"],
                        "text": text,
                    }
                )

            for key in flags:
                flags[key] = 0

    # Write the result to a JSON file in the destination directory, create output folder (yyyy_mm_dd format) if not present in "dest_dir"
    output = {state: {call_sign: {f"{year}-{month}-{day}": result}}}
    output_folder = os.path.join(dest_dir, f"{year}_{month}_{day}")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(dest_dir, f"{year}_{month}_{day}", file_name)

    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
    except IOError:
        logger.error(f"Error writing results to file {output_path}")
        return None

    return output


def process_all_files(source_dir, dest_dir, cache_dir, keyword_dict):
    """
    Start multiple jobs in parallel for sentiment analysis.
    """

    # Load the tokenizer and model with specified cache directory
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest", cache_dir=cache_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest", cache_dir=cache_dir
    )

    # Load the sentiment-analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )

    # Get all files unprocessed transcript files in the source directory
    unprocessed_file_list = get_unprocessed_files(dest_dir, source_dir)

    if len(unprocessed_file_list) == 0:
        logger.info("no files pending sentiment analysis, exiting")
        return

    logger.info(f"total files to process: {len(unprocessed_file_list)}")
    # Process files one at a time
    for i in tqdm(range(len(unprocessed_file_list)), desc="Processing files...."):
        file_name = unprocessed_file_list[i]
        result = sentiment_analysis(
            file_name, source_dir, dest_dir, sentiment_pipeline, keyword_dict
        )


if __name__ == "__main__":
    base_dir = "assets/analytics/sentiment_analysis"

    os.makedirs(base_dir, exist_ok=True)

    source_dir = os.path.join(base_dir, "transcripts")
    dest_dir = os.path.join(base_dir, "sentiment_buffer")
    log_path = os.path.join(base_dir, "logs")
    cache_dir = os.path.join(base_dir, "models")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    logger.add(f"{log_path}/sentiment_analysis.log")

    process_all_files(source_dir, dest_dir, cache_dir, keyword_dict)
