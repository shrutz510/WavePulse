import json
import argparse
from collections import Counter
from typing import List
from loguru import logger


def format_json(input_filename: str) -> str:
    """
    Formats the input JSON file by filtering out items where any key has a null value and
    writes the formatted data to a new JSON file with '_formatted' appended to the original filename.

    Parameters:
        input_filename (str): The name of the input JSON file.

    Returns:
        str: The name of the output formatted JSON file.
    """

    output_filename = input_filename.replace(".json", "_formatted.json")

    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    filtered_data = [item for item in data]

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(
            json.dumps(filtered_data, indent=4, ensure_ascii=False, sort_keys=True)
        )

    logger.info(f"\n{output_filename} created\n")
    return output_filename


def count_rulings(filename: str) -> None:
    """
    Counts the occurrences of each ruling in the formatted JSON file and prints the results.

    Parameters:
        filename (str): The name of the formatted JSON file.

    Example:
        >>> python3 beautify_json.py snopes.json
        >>> snopes_formatted.json created

            ================================================================================
            Total number of entries: 935
            ================================================================================
            Count of each ruling:
                Labeled Satire: 151
                True: 211
                Unfounded: 26
                False: 182
                Misattributed: 18
                Correct Attribution: 56
                Fake: 86
                Miscaptioned: 60
                Mostly True: 10
                Mixture: 58
                Unproven: 35
                Research In Progress: 13
                Originated as Satire: 29

        >>> snopes_formatted.json
    """

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)  # load the formatted data

    rulings = [
        item["ruling"] for item in data if "ruling" in item
    ]  # extract the rulings

    total_entries = len(rulings)  # print the total number of entries
    logger.info(f"Total number of entries: {total_entries}")

    logger.info("Count of each ruling:")  # print the count of each ruling
    for ruling, count in total_entries.items():
        logger.info(f"\t{ruling}: {count}")


def main() -> None:
    """
    Modify JSON to make it more readable and get a count of rulings.

    Example:
        >>> python3 beautify_json.py politifact_2024.json
        >>> main()
    """

    parser = argparse.ArgumentParser(
        description="Process and count rulings in a JSON file."
    )
    parser.add_argument(
        "input_filename", type=str, help="The input JSON file to be formatted."
    )
    args = parser.parse_args()

    formatted_filename = format_json(args.input_filename)
    count_rulings(formatted_filename)

    return None


if __name__ == "__main__":
    main()
