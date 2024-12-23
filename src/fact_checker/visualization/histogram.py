import json
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict
from loguru import logger


def extract_and_count_rulings(data: List[Dict]) -> Dict[str, int]:
    """
    Extracts the 'ruling-unified' attribute from a list of dictionaries and counts the frequency of each ruling.

    Parameters:
        data (List[Dict]): A list of dictionaries containing the JSON data.

    Returns:
        Dict[str, int]: A dictionary with ruling counts.

    Example:
        >>> ruling_counts = extract_and_count_rulings(data)
        {'False': 1833, 'Miscaptioned': 137, 'Satire': 324, 'Unverified': 137,
        'True': 469, 'Mixed': 132, 'Mostly True': 45, 'Mostly False': 51, 'Scam': 2,
        'Outdated': 11, 'Only Analysis, No Label': 893, 'Misleading': 139}
    """

    ruling_counts = {}

    for item in data:
        if isinstance(item, dict) and "ruling-unified" in item:
            ruling = item["ruling-unified"]
            if ruling in ruling_counts:
                ruling_counts[ruling] += 1
            else:
                ruling_counts[ruling] = 1

    return ruling_counts


def create_bar_plot(ruling_counts: Dict[str, int]) -> None:
    """
    Creates a horizontal bar plot showing the frequency of each ruling.

    Parameters:
        ruling_counts (Dict[str, int]): A dictionary with ruling counts.
        desired_order (List[str]): A list of rulings in the desired order.
        custom_texts (Dict[str, str]): A dictionary with custom texts for each ruling.

    Example:
        >>> create_bar_plot(ruling_counts, desired_order, custom_texts)
        histogram.png
    """

    desired_order = [
        "Only Analysis, No Label",
        "Outdated",  # Scam,
        "Satire",
        "Miscaptioned",
        "Misleading",
        "Decontextualized",
        "False",
        "Mostly False",
        "Mixed",
        "Mostly True",
        "True",
    ]  # define the desired order of rulings

    custom_texts = {
        "Only Analysis, No Label": "Biden Did NOT Say There's A $2,880 Medicare Flex Card For Seniors",
        "Outdated": "Is It True That No One Has Died with Omicron Variant?",
        # 'Scam': "MrBeast Launched Casino App 'The Beast Plinko' with Endorsements from Andrew Tate and The Rock?",
        "Satire": "Trump Rejected Female Jurors for Not Being 'His Type'?",
        "Miscaptioned": "Does Video Show Biden Arriving in Israel in October 2023?",
        "Misleading": "Post About SCOTUS Ruling On SB 4 Is Misleading And Outdated",
        "Decontextualized": "Kentucky Cutting Polling Places",
        "False": "Real Photo of Trump Serving in the Military?",
        "Mostly False": "Valid IDs are not required for voting.",
        "Mixed": "NYC will give $1,000 taxpayer-funded credit cards to migrants.",
        "Mostly True": "Under President Joe Biden, “Black unemployment is the lowest in American history.",
        "True": "Donald Trump “deported less, believe it or not, than Barack Obama even did.",
    }  # example texts for each ruling

    colors = {
        "Only Analysis, No Label": "#FF00FF",
        "Outdated": "#FF00FF",
        # 'Scam': '#FF00FF',
        "Satire": "#FF00FF",
        "Miscaptioned": "#FF00FF",
        "Misleading": "#FF00FF",
        "Decontextualized": "#FF00FF",
        "False": "#0000FF",
        "Mostly False": "#3399FF",
        "Mixed": "#99CCFF",
        "Mostly True": "#FFCC33",
        "True": "#FFFF00",
        "": "#FFFFFF",  # color for the gap
    }  # define colors for each ruling

    bar_colors = [
        colors.get(ruling, "#FFFFFF") for ruling in desired_order
    ]  # default to white if ruling not found
    counts = [ruling_counts.get(ruling, 0) for ruling in desired_order]
    gap_index = desired_order.index("False")  # insert gap without affecting bar_colors
    counts.insert(gap_index, 0)
    desired_order.insert(gap_index, "")
    bar_colors.insert(gap_index, "#FFFFFF")  # insert white color for the gap

    plt.figure(figsize=(14, 8))
    plt.barh(
        range(len(desired_order)), counts, color=bar_colors, edgecolor="black"
    )  # create the bar plot with gaps
    plt.xlabel("Frequency", fontsize=14)  # ensure axis titles are displayed
    plt.ylabel("Ruling", fontsize=14)
    plt.title("Frequency of Rulings", fontsize=16)
    plt.xticks(
        range(0, int(max(counts)) + 500, 500), fontsize=10
    )  # add numbers to x-axis at every 500 units
    plt.yticks(range(len(desired_order)), desired_order, fontsize=10)

    for i, (ruling, count) in enumerate(
        zip(desired_order, counts)
    ):  # label the exact number of rulings on the plot and add custom text
        if ruling:  # only add labels for counts greater than 0
            plt.text(count + 0.1, i, int(count), va="center", fontsize=10)
            if ruling in custom_texts:
                plt.text(
                    count + 350,
                    i,
                    custom_texts[ruling],
                    va="center",
                    fontsize=10,
                    color="black",
                )

    plt.tight_layout()  # adjust layout to ensure everything fits well
    plt.savefig("histogram.png")  # save and show plot
    plt.show()

    logger.info(f"Histogram saved as histogram.png'")

    return None


def main() -> None:
    """
    Main function to load JSON data, extract and count rulings, and create a bar plot.

    Example:
        >>> python3 histogram.py new_merged_output_2024.json
        >>> main()
    """

    parser = argparse.ArgumentParser(description="Histogram of data.")
    parser.add_argument("file", type=str, help="The JSON file to create histogram.")
    args = parser.parse_args()

    input_file = f"{args.file}"  # input file name
    with open(input_file, "r") as file:
        data = json.load(file)  # load input file

    ruling_counts = extract_and_count_rulings(data)
    create_bar_plot(ruling_counts)

    return None


if __name__ == "__main__":
    main()
