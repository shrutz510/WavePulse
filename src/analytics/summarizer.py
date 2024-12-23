import argparse
import json
import os
import time
import random
import warnings

from functools import wraps
from tqdm import tqdm


warnings.simplefilter(action="ignore", category=FutureWarning)


import google.generativeai as genai

from google.generativeai.types import HarmCategory, HarmBlockThreshold

api_key = str(os.environ["GCP_API_KEY"])
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.5,
    "top_p": 0.90,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
platform = "gcp_gemini"
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def get_summarization_prompt(new_content, prior_summary):
    summarization_prompt = f"""You are a concise and direct news summarizer. Given below is a JSON with spoken text and its speaker ID recorded from a radio livestream. Create a summary that:

    1. Presents information directly, without phrases like "I heard" or "The news reported."
    2. Uses a factual, journalistic tone as if directly reporting the news.
    3. Retains key facts and information while making the content specific and granular.
    4. Removes personal identifiable information (PII), such as phone numbers and sensitive personal data, but keeps public figures' names (e.g., politicians, celebrities) and other key proper nouns relevant to the context.
    5. Is clear and avoids vague language.
    6. Clarifies ambiguous words or phrases.
    7. Utilizes changes in speaker ID to understand the flow of conversation or different segments of news.
    8. Corresponds strictly to information derived from the provided text.
    9. Organizes information into coherent paragraphs, each focusing on a distinct topic or news item.
    10. Maintains a neutral, objective tone throughout the summary.

    Do not include any meta-commentary about the summarization process or the source of the information.

    Spoken Text Transcription: {new_content} 
    """
    # You can also use the previous summary for context, but ignore it if it is unrelated: {prior_summary}
    return summarization_prompt


def query_gcp_gemini_api(new_content, prior_summary):
    # try:
    response = model.generate_content(
        get_summarization_prompt(new_content, prior_summary),
        safety_settings=safety_settings,
    )
    response_dict = json.loads(response.text)
    # except:
    return response_dict


def read_json(json_file):
    """
    Extracts conversation entries from a JSON file.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading or processing JSON file {json_file}: {e}")
        return None


def retry_with_fixed_wait(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop("retries", 5)
        max_wait = 2  # Maximum wait time in seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(max_wait, random.uniform(1, max_wait))
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)

    return wrapper


# @retry_with_fixed_wait
def generate_gemini_summary(new_content, prior_summary, retries, user):
    """
    Generate a summary using the Google Gemini API for a given text.
    Implements retry logic with a fixed maximum wait time.
    """
    global platform
    try:
        result = query_gcp_gemini_api(new_content, prior_summary)
        summary = result["summary"]
    except Exception as e:
        print(f"Exception: {e}")
        # print(f"Error in generating summary: {result['error']}")
        return None
    return summary


def save_segments_to_file(summary, output_folder, filename_prefix):
    """
    Save conversation segments and their summaries to files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    segment_filename = f"{filename_prefix}.txt"
    segment_path = os.path.join(output_folder, segment_filename)

    try:
        with open(segment_path, "w", encoding="utf-8") as f:
            f.write(summary)
            f.close()
    except Exception as e:
        print(f"Error saving to {segment_path}: {e}")


def summarize_transcripts(input_folder, output_folder, user):
    """
    Processes conversation files, dynamically merges segments based on similarity, and limits processing to 20 files.
    """

    start_time = time.time()
    transcript_jsons = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".json")]
    )
    prior_summary = "None"  # Hardcoded disabled
    with tqdm(total=len(transcript_jsons), desc="Processing Transcripts") as pbar:
        for filename in transcript_jsons:
            filename_prefix = os.path.splitext(filename)[0]
            if os.path.isfile(os.path.join(output_folder, f"{filename_prefix}.txt")):
                pbar.update(1)
                continue
            # print(f"Processing conversation file: {filename}")
            json_file_path = os.path.join(input_folder, filename)

            transcript = read_json(json_file_path)
            if transcript is None:
                continue

            summary = generate_gemini_summary(
                transcript, prior_summary, retries=5, user=user
            )
            if summary is not None:
                save_segments_to_file(summary, output_folder, filename_prefix)
                print("Processed Successfully: ", filename_prefix)
            else:
                print(f"Failed {json_file_path} after multiple retries")
            pbar.update(1)

    total_time = time.time() - start_time
    print(
        f"Total time to process and summarize conversations: {total_time:.4f} seconds."
    )


# Main function to run the process
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Summarize transcripts from input folder."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input folder containing transcripts",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output folder for summaries"
    )
    parser.add_argument(
        "--use-gcp", action="store_true", help="Path to the output folder for summaries"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.environ["USER"],
        help="Specify User for API access.",
    )
    # Parse arguments
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    user = args.user

    print(f"Starting to process transcripts from {input_folder}")
    summarize_transcripts(input_folder, output_folder, user)
    print("Completed summarizing all conversations.")


if __name__ == "__main__":
    main()
