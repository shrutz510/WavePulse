import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import os
import argparse
from typing import Dict, List
from loguru import logger

# Use Agg backend for matplotlib
import matplotlib

matplotlib.use("Agg")


def extract_titles_by_ruling(merged_data: List[Dict]) -> Dict[str, List[str]]:
    """
    Extracts titles from the JSON data and categorizes them by 'ruling-unified'.

    Parameters:
        merged_data (List[Dict]): The merged JSON data.

    Returns:
        Dict[str, List[str]]: A dictionary with 'ruling-unified' as keys and lists of titles as values.

    Example:
        >>> titles = extract_titles_by_ruling(data)
        {'True': ["Video shows presidential candidate speaking."], 'False': ["The sky is green."],
    """

    titles_by_ruling = {
        "True": [],
        "False": [],
        "Mostly True": [],
        "Mixed": [],
        "Mostly False": [],
        "Remaining": [],
    }
    for item in merged_data:
        ruling = item.get("ruling-unified", "Unknown")
        if ruling in ["True"]:
            titles_by_ruling["True"].append(item["title"])
        elif ruling in ["False"]:
            titles_by_ruling["False"].append(item["title"])
        elif ruling in ["Mostly True", "Mixed", "Mostly False"]:
            titles_by_ruling["Mostly True"].append(item["title"])
            titles_by_ruling["Mixed"].append(item["title"])
            titles_by_ruling["Mostly False"].append(item["title"])
        else:
            titles_by_ruling["Remaining"].append(item["title"])

    return titles_by_ruling


def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by converting to lowercase, removing specific phrases,
    special characters, and single-letter words.

    Parameters:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.

    Example:
        >>> clean_text = preprocess_text("Video shows presidential candidate speaking.")
        "Video shows presidential candidate speaking."
    """

    text = text.lower()
    text = re.sub(
        r"(?i)\b(video) (shows)\b", r'\1 "\2"', text
    )  # separate 'video shows' into two words
    text = re.sub(
        r"(?i)\b(video) (show)\b", r'\1 "\2"', text
    )  # separate 'video show' into two words
    text = re.sub(
        r"(?i)\b(donald) (trump)\b", r'\1 "\2"', text
    )  # separate 'donald trump' into two words
    text = re.sub(
        r"(?i)\b(joe) (biden)\b", r'\1 "\2"', text
    )  # separate 'joe biden' into two words
    text = re.sub(
        r"[^a-z\s]", "", text
    )  # remove special characters, keeping only letters and spaces
    text = " ".join(
        word for word in text.split() if len(word) > 1
    )  # exclude single-letter words

    return text


def generate_word_clouds(titles_by_ruling: Dict[str, List[str]]) -> None:
    """
    Generates word clouds for each category of titles and saves them as image files.

    Parameters:
        titles_by_ruling (Dict[str, List[str]]): A dictionary with categories as keys and lists of titles as values.

    Example:
        >>> generate_word_clouds(titles_by_ruling)
        Word cloud saved as 'wordclouds/wordcloud_True.png'
        Word cloud saved as 'wordclouds/wordcloud_False.png'
        Word cloud saved as 'wordclouds/wordcloud_Middle.png'
        Word cloud saved as 'wordclouds/wordcloud_Miscellaneous.png'
    """

    stopwords = set(STOPWORDS)
    categories = {
        "True": titles_by_ruling["True"],
        "False": titles_by_ruling["False"],
        "Middle": titles_by_ruling["Mostly True"]
        + titles_by_ruling["Mixed"]
        + titles_by_ruling["Mostly False"],
        "Miscellaneous": titles_by_ruling["Remaining"],
    }

    for category, titles in categories.items():
        if titles:
            text = " ".join(titles)
            text = preprocess_text(text)
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", stopwords=stopwords
            ).generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")

            output_dir = "wordclouds"  # ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(
                output_dir, f'wordcloud_{category.replace(" ", "_")}.png'
            )
            plt.savefig(output_path)  # save the word cloud as an image file
            logger.info(f"Word cloud saved as '{output_path}'")

    return None


def main() -> None:
    """
    Main function to load merged JSON data, extract titles by ruling, and generate word clouds.

    Example:
        >>> python3 word_cloud.py new_merged_output_2024.json
        >>> main()
    """

    parser = argparse.ArgumentParser(description="Word cloud of data.")
    parser.add_argument("file", type=str, help="The JSON file to create word cloud.")
    args = parser.parse_args()

    input_file = f"{args.file}"  # input file name
    with open(input_file, "r") as file:
        merged_data = json.load(file)  # load input file

    titles_by_ruling = extract_titles_by_ruling(merged_data)
    generate_word_clouds(titles_by_ruling)

    return None


if __name__ == "__main__":
    main()
