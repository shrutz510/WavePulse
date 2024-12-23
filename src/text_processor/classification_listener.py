"""
Create listener to check and trigger classification for any transcript files present/pending transcription in unclassified buffer folder.
After classification save the transcript in the following format in .txt file:
    24/06/2024, 23:02:16 - SPEAKER_01: Attorney General and not Congress.
    24/06/2024, 23:02:18 - SPEAKER_02: The trial itself is indefinitely postponed.
"""

import config
import contextlib

import os
import json
import time

import concurrent.futures

from text_processor.classifier import classify_transcript
from args import get_args
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict
from utils.timezone_converter import convert_timezone


def reformat_and_save(
    audio_file_name: str,
    political_output_txt_file: str,
    political_ad_output_txt_file: str,
    apolitical_output_txt_file: str,
    classified_transcript: List[Dict],
    diarize: bool,
) -> None:
    """
    Reformat audio transcript into following format and save as .txt file:
        24/06/2024, 23:02:16 - SPEAKER_01: Attorney General and not Congress.
        24/06/2024, 23:02:18 - SPEAKER_01: The trial itself is indefinitely postponed.
    """
    audio_file_name = audio_file_name.split(".")[0]
    name_details = audio_file_name.split("_")
    state_code = name_details[0]
    year = int(name_details[-5])
    month = int(name_details[-4])
    day = int(name_details[-3])
    hour = int(name_details[-2])
    minute = int(name_details[-1])
    rec_start_time = datetime(year, month, day, hour, minute)

    rec_start_time = convert_timezone(rec_start_time, "NY", state_code)
    # flag to store if apolitical segments or advertisement segments are being written written. At a time only one will be on.
    apolitical_flag = 0
    ad_flag = 0
    political_flag = 0
    logger.info(
        f"opening text files: {political_output_txt_file}, {political_ad_output_txt_file}, {apolitical_output_txt_file}"
    )
    with open(political_output_txt_file, "w") as file1, open(
        political_ad_output_txt_file, "w"
    ) as file2, open(apolitical_output_txt_file, "w") as file3:
        logger.info(f"writing txt file for: {audio_file_name}")
        for segment in classified_transcript:
            timestamp = rec_start_time + timedelta(seconds=segment["start"])
            formatted_timestamp = timestamp.strftime("%d/%m/%Y, %H:%M:%S")
            if diarize:
                speaker = segment.get("speaker", "unknown")
            else:
                speaker = "na"

            if (
                segment["content_class"] == "Apolitical Content"
                and apolitical_flag == 0
            ):
                file1.write(f"Apolitical Content .................\n")
                file2.write(f"Apolitical Content .................\n")
                file3.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')
                apolitical_flag = 1
                ad_flag = 0
                political_flag = 0
                continue
            elif (
                segment["content_class"] == "Apolitical Content"
                and apolitical_flag == 1
            ):
                file3.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')
                continue
            elif segment["ad_class"] == "Advertisement" and ad_flag == 0:
                file1.write(f"Political Advertisement .................\n")
                file2.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')
                file3.write(f"Political Advertisement .................\n")
                apolitical_flag = 0
                ad_flag = 1
                political_flag = 0
                continue
            elif segment["ad_class"] == "Advertisement" and ad_flag == 1:
                file2.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')
                continue
            elif political_flag == 0:
                apolitical_flag = 0
                ad_flag = 0
                political_flag = 1
                file1.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')
                file2.write(f"Political Content .................\n")
                file3.write(f"Political Content .................\n")
            else:
                file1.write(f'{formatted_timestamp} - {speaker}: {segment["text"]}\n')

    logger.info(f"text file written for {audio_file_name}")
    return


def start_classification_listener(transcripts_dir: str):
    """
    Create and start a listener in background that periodically checks for transcripts pending classification in unclassified_buffer folder.
    """
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        executor.map(classification_listener, [transcripts_dir])
        logger.info(f"starting classification listener in background successfull ...")

    except (KeyboardInterrupt, SystemExit):
        logger.info(f"stopped classification listener ...")
    return executor


def classification_listener(transcripts_dir: str) -> None:
    """
    Listener that monitors the unclassified_buffer for transcripts pending classification.
    It triggers classification for multiple transcripts in parallel.
    Classifies transcript content into Political/Apolitical content, then Political Content into Advertisement/Not Advertisement.
    Classification is done using Google's gemini.
    """
    args = get_args()
    logger.info("starting classification listener .....")
    temp_file_dir = os.path.join(transcripts_dir, "unclassified_buffer")
    classified_file_dir = os.path.join(transcripts_dir, "classified")
    classified_json_file_dir = os.path.join(classified_file_dir, "json")
    classified_political_file_dir = os.path.join(classified_file_dir, "political")
    classified_political_ad_file_dir = os.path.join(classified_file_dir, "political_ad")
    classified_apolitical_file_dir = os.path.join(classified_file_dir, "apolitical")

    try:
        while config.shared_config["running"]:
            # files to be classified in one batch, default is 10 (based on gemini api rate limit)
            n = args.concurrent_classification
            logger.info(
                f"checking for transcript files pending classification in {temp_file_dir}"
            )
            temp_files = os.listdir(temp_file_dir)
            for file in temp_files:
                if not file.endswith(".json"):
                    temp_files.remove(file)
            if not temp_files:
                logger.info(f"no new transcripts found to be classified")
                time.sleep(60)
                continue
            elif len(temp_files) < n:
                n = len(temp_files)

            logger.info(
                f"{len(temp_files)} files found pending classification, classifying for {n} files"
            )
            temp_files = temp_files[:n]
            transcript_list = []
            for i in range(n):
                file = temp_files[i]
                temp_file = os.path.join(temp_file_dir, file)
                f = open(temp_file)
                transcript_list.append(json.load(f))

            # Classify content of multiple transcripts in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        classify_transcript, transcript_list[i], temp_files[i]
                    )
                    for i in range(n)
                ]
                classified_transcripts = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

            logger.info(f"got gemini response for {n} transcripts")
            for i in range(n):
                logger.info(f"finalizing i: {i}")
                result_dict = classified_transcripts[i]
                file = result_dict["file_name"]
                logger.info(f"finalizing file: {file}")
                classified_json_file = os.path.join(classified_json_file_dir, file)
                classified_transcript = result_dict["transcript"]
                logger.info(f"classified_json_file: {classified_json_file}")
                with open(classified_json_file, "w") as json_file:
                    json.dump(classified_transcript, json_file, indent=4)

                political_output_txt_file = os.path.join(
                    classified_political_file_dir, f"{file.split('.')[0]}.txt"
                )
                political_ad_output_txt_file = os.path.join(
                    classified_political_ad_file_dir, f"{file.split('.')[0]}.txt"
                )
                apolitical_output_txt_file = os.path.join(
                    classified_apolitical_file_dir, f"{file.split('.')[0]}.txt"
                )

                diarize = False
                if "speaker" in classified_transcript[0]:
                    diarize = True
                else:
                    diarize = False
                logger.info(f"saving as text file")
                reformat_and_save(
                    file,
                    political_output_txt_file,
                    political_ad_output_txt_file,
                    apolitical_output_txt_file,
                    classified_transcript,
                    diarize,
                )
                logger.debug(f"deleting temp file ...")
                temp_file = os.path.join(temp_file_dir, file)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(temp_file)
            logger.info(f"classified for {n} files successfully!")
        logger.info(
            f"stopping classification listener for buffer: {temp_file_dir} as config.shared_config['running']: {config.shared_config['running']}"
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info(f"stopped classification listener")
    logger.info("exited classification listener .....")
    return
