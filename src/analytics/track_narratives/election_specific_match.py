import h5py
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import argparse
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Load document paths
def load_document_paths(file_path: str) -> List[str]:
    with h5py.File(file_path, "r") as f:
        filepaths = [path.decode("utf-8") for path in f["filepaths"][:]]
    return filepaths


# Load LLM
def load_llm(model_name: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    # Fix for the padding token warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model, device


def parse_model_output(response: str) -> Dict:
    # Try to find a JSON-like structure in the response
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If JSON parsing fails, try to extract information manually
    mention_count = 0
    stance = "unknown"

    # Look for mention count
    count_match = re.search(r'mention_count"?\s*:\s*(\d+)', response)
    if count_match:
        mention_count = int(count_match.group(1))

    if mention_count == 0:
        return None

    # Look for stance
    stance_match = re.search(
        r'stance"?\s*:\s*"?(support|vanilla|debunk)', response, re.IGNORECASE
    )
    if stance_match:
        stance = stance_match.group(1).lower()

    return {"mention_count": mention_count, "stance": stance}


def analyze_batch(batch_content: List[str], tokenizer, model, device) -> List[Dict]:
    prompt_template = """
    Analyze the following document summary regarding mentions of the 2020 election being stolen, rigged, or false.

    Document summary:
    {content}

    Answer the following questions:
    1. How many times was the 2020 election being stolen, rigged, or false mentioned?
    2. Did the document support, vanilla report, or debunk these claims?

    Provide your answer in the following format:
        "mention_count": <number of mentions>,
        "stance": "<support/vanilla/debunk>"
    """

    prompts = [prompt_template.format(content=content) for content in batch_content]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    results = []
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for response in responses:
            result = parse_model_output(response)
            if result is not None:
                results.append(result)
    except Exception as e:
        logging.error(f"Error in model generation or parsing: {str(e)}")
        results = [{"mention_count": 0, "stance": "error"} for _ in batch_content]

    return results


def analyze_dataset(
    filepaths: List[str], root_path: str, model_name: str, batch_size: int
):
    tokenizer, model, device = load_llm(model_name)
    results = []

    for i in tqdm(range(0, len(filepaths), batch_size), desc="Processing batches"):
        batch_paths = filepaths[i : i + batch_size]
        batch_content = []
        valid_paths = []

        for filepath in batch_paths:
            full_path = os.path.join(root_path, filepath.split("_")[1], filepath)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    batch_content.append(content)
                    valid_paths.append(filepath)
                except Exception as e:
                    logging.error(f"Error reading file {full_path}: {str(e)}")
            else:
                logging.warning(f"File not found: {full_path}")

        if batch_content:
            batch_results = analyze_batch(batch_content, tokenizer, model, device)
            for filepath, analysis in zip(valid_paths, batch_results):
                results.append({"filepath": filepath, "analysis": analysis})

    return results


# Save results
def save_results(results: List[Dict], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze 2020 Election Mentions in Documents"
    )
    parser.add_argument("-i", "--input", required=True, help="the H5 file to process.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing documents",
    )
    args = parser.parse_args()

    embeddings_file_idx = int(args.input)
    summary_root_path = "/vast/gm2724/transcripts_summarized/"
    root_path = "/scratch/gm2724/radio-observatory-dev/rag_test/"
    embeddings_file = str(
        list(Path(root_path).glob("embeddings_*.h5"))[embeddings_file_idx].name
    )
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    call_sign = embeddings_file[11:-3]
    output_file = f"election_2020_{call_sign}.json"
    print(f"Doing File: {embeddings_file}")

    filepaths = load_document_paths(embeddings_file)
    results = analyze_dataset(filepaths, summary_root_path, model_name, args.batch_size)
    save_results(results, output_file)

    print(f"Analysis complete. Results saved to {output_file}")
