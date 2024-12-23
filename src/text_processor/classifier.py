"""
Summarize and classify the contents of transcribed file using gemini into one of 3 categories:
 1. Political News/ Discussion
 2. Political Ad
 3. Apolitical Content

 30 min transcript is divided into 3 equal parts for sending as input to gemini. 
This is for reducing the output length of gemini to prevent getting truncated outputs.
"""

import os
import json
import time

import concurrent.futures

from args import get_args
from text_processor import prompts
from loguru import logger
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Any


def classify_transcript(transcript: str, file_name: str):
    """
    Setup gemini model hyperparameters, then classify transcript content into Political/Apolitical content and Ad/Not Ad content.
    Divide 30 min transcript content into 3 equal parts, before classifying, to avoid getting truncated output due to long output length.
    """
    logger.info(f"loading gemini model")
    args = get_args()
    api_key = args.gemini_api_key
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0,
        "top_p": 0.90,
        "top_k": 50,
        "max_output_tokens": 10000,
        "response_mime_type": "application/json",
    }
    logger.debug(f"gemini model config: {generation_config}")

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    model.start_chat()

    logger.info("gemini model loaded successfully")
    n = len(transcript)
    k = 3
    if n <= k:
        transcript_parts = [transcript]
    else:
        part_size = n // k
        transcript_parts = [
            transcript[i * part_size : (i + 1) * part_size] for i in range(k - 1)
        ]
        transcript_parts.append(transcript[(k - 1) * part_size :])

    logger.info(
        f"calling classify_political for {file_name}: {len(transcript_parts)} parts"
    )
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures_political = [
            executor.submit(
                classify_political, model, transcript_parts[part_index], part_index
            )
            for part_index in range(len(transcript_parts))
        ]
        done, not_done = concurrent.futures.wait(
            futures_political,
            timeout=None,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        political_classification_result = [
            future.result()["transcript_part"] for future in done
        ]
        logger.info(f"political classification done, starting ad classification ....")
        futures_ad = [
            executor.submit(
                classify_ad,
                model,
                political_classification_result[part_index],
                part_index,
            )
            for part_index in range(len(political_classification_result))
        ]
        ad_done, ad_not_done = concurrent.futures.wait(
            futures_ad, timeout=None, return_when=concurrent.futures.ALL_COMPLETED
        )
        final_classified_parts = [future.result() for future in ad_done]
    end_time = time.time()

    classified_transcript = []
    for classified_part in final_classified_parts:
        classified_transcript.extend(classified_part["transcript_part"])
    classified_transcript = sorted(classified_transcript, key=lambda x: x["start"])

    total_time = end_time - start_time
    logger.info(f"file {file_name} classified in: {total_time:.2f}s")

    response = {"file_name": file_name, "transcript": classified_transcript}
    return response


def classify_ad(
    model, transcript_part: List[Dict[str, Any]], part_index: int
) -> Dict[str, Any]:
    """
    Classify transcript content (segment by segment) into Advertisement or Not Advertisement.
    Adds "ad_class" key for each segment/sentence with appropriate classification label.
    Parameters:
        - "model": gemini model
        - "transcript_part" list(dict): Transcript in json format. List of dictionary of following elements
            - "start" (float): start time of text segment
            - "end" (float): end time of text segment
            - "text" (str): text segment
            - "speaker" (str): speaker id
        - "part_index" (int): index to maintain relative order of parts of transcript (transcript gets split in 3 parts to manage output length).
    Returns:
        - "transcript_part" list(dict): modified transcript with additional key of "ad_class" in each text segment
    Example:
    >>> transcript_part
    [
        {
            "start": 0.309,
            "end": 8.351,
            "text": "Free free free, grab your free copy of Death Stranding from epic games store now.",
            "speaker": "SPEAKER_00"
        },
        {
            "start": 10.052,
            "end": 25.536,
            "text": "Young people on many college campuses, from the East Coast at Columbia to the West Coast at UCLA,
                overwhelmingly set up protests on campus to say that the Palestinians were the force for good",
            "speaker": "SPEAKER_01"
        }
    ]
    >>> classify_ad(model, transcript_part, part_index)
    [
        {
            "start": 0.309,
            "end": 8.351,
            "text": "Free free free, grab your free copy of Death Stranding from epic games store now.",
            "speaker": "SPEAKER_00",
            "ad_class": "Advertisement"
        },
        {
            "start": 10.052,
            "end": 25.536,
            "text": "Young people on many college campuses, from the East Coast at Columbia to the West Coast at UCLA,
                overwhelmingly set up protests on campus to say that the Palestinians were the force for good",
            "speaker": "SPEAKER_01",
            "ad_class": "Not Advertisement"
        }
    ]
    """
    # logger.info(transcript_part)
    concatenated_text = "".join(seg["text"] for seg in transcript_part)
    # logger.info(f"input text: {concatenated_text}")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    prompt = [
        prompts.ad_classification_que,
        f"input: {prompts.ad_classification_input}",
        f"output: {prompts.ad_classification_output}",
        f"input: {concatenated_text}",
        "output: ",
    ]
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        # logger.info(f"got gemini response for part_index: {part_index}")
        response_dict = json.loads(response.text)
        # logger.info(f"loaded gemini response as dict for part_index: {part_index}")
        # logger.info(f"length of response_dict: {len(response_dict)}")

        for seg in transcript_part:
            text = seg["text"].strip()
            seg["ad_class"] = None
            for key, value in response_dict.items():
                if text in key:
                    seg["ad_class"] = value

            if not seg["ad_class"]:
                seg["ad_class"] = "unsure"

        # logger.info(f"length of transcript_part: {len(transcript_part)}")

        # return transcript_part
    except (ValueError):
        logger.error(f"got value error in following output: {response}")
        logger.error(f"input: {concatenated_text}")
        for seg in transcript_part:
            text = seg["text"].strip()
            seg["ad_class"] = "unsure due to error"
        logger.error(f"exiting classify_ad after error")
        # logger.info(f"length of transcript_part: {len(transcript_part)}")

        # return transcript_part
    finally:
        result = {"part_index": part_index, "transcript_part": transcript_part}
        return result


def classify_political(
    model, transcript_part: List[Dict[str, Any]], part_index: int
) -> Dict[str, Any]:
    """
    Classify transcript content (segment by segment) into Political or Apolitical Content.
    Adds "content_class" key for each segment/sentence with appropriate classification label.

    Parameters:
        - "model": gemini model
        - "transcript_part" list(dict): Transcript in json format. List of dictionary of following elements
            - "start" (float): start time of text segment
            - "end" (float): end time of text segment
            - "text" (str): text segment
            - "speaker" (str): speaker id
        - "part_index" (int): index to maintain relative order of parts of transcript (transcript gets split in 3 parts to manage output length).
    Returns:
        - "transcript_part" list(dict): modified transcript with additional key of "content_class" in each text segment

    Example:
    >>> transcript_part
    [
        {
            "start": 0.309,
            "end": 8.351,
            "text": "Free free free, grab your free copy of Death Stranding from epic games store now.",
            "speaker": "SPEAKER_00"
        },
        {
            "start": 10.052,
            "end": 25.536,
            "text": "Young people on many college campuses, from the East Coast at Columbia to the West Coast at UCLA,
                overwhelmingly set up protests on campus to say that the Palestinians were the force for good",
            "speaker": "SPEAKER_01"
        }
    ]
    >>> classify_ad(model, transcript_part, part_index)
    [
        {
            "start": 0.309,
            "end": 8.351,
            "text": "Free free free, grab your free copy of Death Stranding from epic games store now.",
            "speaker": "SPEAKER_00",
            "content_class": "Apolitical Content"
        },
        {
            "start": 10.052,
            "end": 25.536,
            "text": "Young people on many college campuses, from the East Coast at Columbia to the West Coast at UCLA,
                overwhelmingly set up protests on campus to say that the Palestinians were the force for good",
            "speaker": "SPEAKER_01",
            "content_class": "Political Content"
        }
    ]
    """
    concatenated_text = "".join(seg["text"] for seg in transcript_part)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    prompt = [
        prompts.political_classification_que,
        f"input: {prompts.political_classification_input}",
        f"output: {prompts.political_classification_output}",
        f"input: {concatenated_text}",
        "output: ",
    ]
    # logger.info(f"starting classify_political for part_index: {part_index}")
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        # logger.info(f"got gemini response for part_index: {part_index.}")
        response_dict = json.loads(response.text)
        # logger.info(f"loaded gemini response as dict for part_index: {part_index}")
        # logger.info(f"length of response_dict: {len(response_dict)}")

        for seg in transcript_part:
            text = seg["text"].strip()
            seg["content_class"] = None
            for key, value in response_dict.items():
                if text in key:
                    seg["content_class"] = value

            if not seg["content_class"]:
                seg["content_class"] = "unsure"

        # logger.info(f"length of transcript_part: {len(transcript_part)}")

        # return transcript_part
    except (ValueError):
        logger.error(f"got value error in following output: {response}")
        logger.error(f"input: {concatenated_text}")
        for seg in transcript_part:
            text = seg["text"].strip()
            seg["content_class"] = "unsure due to error"
        logger.error(f"exiting classify_political after error")
        # logger.info(f"length of transcript_part: {len(transcript_part)}")

        # return transcript_part
    finally:
        result = {"part_index": part_index, "transcript_part": transcript_part}
        return result


if __name__ == "__main__":
    args = get_args()

    transcripts_dir = os.path.join(args.assets_dir, args.data_dir, args.transcripts_dir)
    transcripts_temp_dir = os.path.join(transcripts_dir, "temp")
    transcripts_political_dir = os.path.join(transcripts_dir, "political")

    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(transcripts_temp_dir, exist_ok=True)
    os.makedirs(transcripts_political_dir, exist_ok=True)

    file = "FL_WHBO_2024_07_09_07_45.json"

    local_path = os.path.join(transcripts_temp_dir, file)
    f = open(local_path)
    result = json.load(f)

    response = classify_transcript(result, file)
    classified_json_file = os.path.join(transcripts_political_dir, file)
    with open(classified_json_file, "w") as json_file:
        json.dump(response["transcript"], json_file, indent=4)
