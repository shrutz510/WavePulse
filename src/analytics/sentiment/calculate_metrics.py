#!/bin/python

import csv
import json
import os
import pandas as pd
import numpy as np
import subprocess
import sys
import time


from datetime import datetime
from loguru import logger
from tqdm import tqdm


def merge(source_dir, dest_dir, dest_file):
    """
    Merge processed individual json files
    """
    files = os.listdir(source_dir)
    json_files = [file for file in files if file.endswith(".json")]

    result = {}

    for i in tqdm(range(0, len(json_files)), desc="Processing files"):
        file_name = json_files[i]
        local_path = os.path.join(source_dir, file_name)
        f = open(local_path)
        sentiment_analysis = json.load(f)
        for state in sentiment_analysis.keys():
            if state not in result.keys():
                result[state] = {}
            for call_sign in sentiment_analysis[state].keys():
                if call_sign not in result[state].keys():
                    result[state][call_sign] = {}
                for date in sentiment_analysis[state][call_sign].keys():
                    for key, value in sentiment_analysis[state][call_sign][
                        date
                    ].items():
                        if key not in result[state][call_sign].keys():
                            result[state][call_sign][key] = sentiment_analysis[state][
                                call_sign
                            ][date][key]
                        else:
                            result[state][call_sign][key].extend(
                                sentiment_analysis[state][call_sign][date][key]
                            )

    output_path = os.path.join(dest_dir, f"{dest_file}.json")
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
    except IOError:
        logger.error(f"Error writing results to file {output_path}")
        return None
    return result


def list_new_folders(dest_dir, start_date_str):
    """
    Return a list of all folders in the given directory after "start_date", excluding those that start with a '.'
    """
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")

    folders = [
        name
        for name in os.listdir(dest_dir)
        if os.path.isdir(os.path.join(dest_dir, name)) and not name.startswith(".")
    ]
    filtered_folders = []
    for folder in folders:
        parts = folder.split("_")
        folder_date = datetime.strptime(f"{parts[0]}_{parts[1]}_{parts[2]}", "%Y_%m_%d")
        if folder_date >= start_date:
            filtered_folders.append(folder)
    return filtered_folders


def merge_json_files(sentiment_buffer, merged_json, start_date_str="2024_06_26"):
    """
    Merge json files with sentiment analysis data date-wise and save it in "merged_json" folder
    """
    source_folders = list_new_folders(sentiment_buffer, start_date_str)

    dest_dir = merged_json
    os.makedirs(dest_dir, exist_ok=True)
    for folder in source_folders:
        dest_file = folder
        source_dir = os.path.join(sentiment_buffer, folder)
        logger.info(f"processing folder: {dest_file}")
        result = merge(source_dir, dest_dir, dest_file)


def calc_metrics(text, dest_dir, dest_file, keyword_list):
    """
    Calculate metrics and store in a csv
    """
    result = []

    filename = os.path.join(dest_dir, f"{dest_file}.csv")

    fields = ["State", "Call_Sign"]

    for keyword in keyword_list:
        fields.append(f"{keyword}_Positive_Count")
        fields.append(f"{keyword}_Neutral_Count")
        fields.append(f"{keyword}_Negative_Count")

    for state in tqdm(text.keys()):
        for call_sign in text[state].keys():
            csv_line = {"State": state, "Call_Sign": call_sign}

            metrics_dict = {}

            for keyword in keyword_list:
                metrics_dict[keyword] = [0, 0]
                metrics_dict[keyword] = {"positive": 0, "neutral": 0, "negative": 0}
                for seg in text[state][call_sign][keyword]:
                    if seg["label"] == "positive":
                        metrics_dict[keyword]["positive"] += 1
                    elif seg["label"] == "negative":
                        metrics_dict[keyword]["negative"] += 1
                    else:
                        metrics_dict[keyword]["neutral"] += 1
                csv_line[f"{keyword}_Positive_Count"] = metrics_dict[keyword][
                    "positive"
                ]
                csv_line[f"{keyword}_Neutral_Count"] = metrics_dict[keyword]["neutral"]
                csv_line[f"{keyword}_Negative_Count"] = metrics_dict[keyword][
                    "negative"
                ]

            result.append(csv_line)

    with open(filename, "w") as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(result)
    return


def list_new_json_files(source_dir, start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")

    files = os.listdir(source_dir)
    json_files = [file for file in files if file.endswith(".json")]
    filtered_files = []
    for file in json_files:
        file_date_str = file.split(".")[0]
        parts = file_date_str.split("_")
        file_date = datetime.strptime(f"{parts[0]}_{parts[1]}_{parts[2]}", "%Y_%m_%d")
        if file_date >= start_date:
            filtered_files.append(file)
    return filtered_files


def calculate_stats(merged_json_dir, dest_dir, start_date_str="2024_06_26"):
    """
    Create intermediate csv files with per call sign per transcript data
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Get a list of merged json files
    json_files = list_new_json_files(merged_json_dir, start_date_str)

    # Calculate metrics for each json file
    for json_file in json_files:
        local_path = os.path.join(merged_json_dir, json_file)
        f = open(local_path)
        text = json.load(f)

        dest_file = json_file.split(".")[0]
        logger.info(f"calculating metrics for: {dest_file}.csv")
        keyword_list = [
            "Biden",
            "Harris",
            "Trump",
            "Democrats",
            "Republicans",
            "Taylor_Swift",
            "Musk",
            "Olympics",
            "Biden-Trump",
            "Harris-Trump",
            "Democrats-Republicans",
        ]
        calc_metrics(text, dest_dir, dest_file, keyword_list)


def combined_sentiment(source_dir, file_name, column_order, keyword_list):
    """
    Combine sentiment by call-sign for a day
    """
    local_path = os.path.join(source_dir, file_name)
    df = pd.read_csv(local_path)

    for keyword in keyword_list:
        df[f"{keyword}_Count"] = (
            df[f"{keyword}_Positive_Count"]
            + df[f"{keyword}_Neutral_Count"]
            + df[f"{keyword}_Negative_Count"]
        )
        # Calculate the combined sentiment
        df[f"{keyword}_Combined_Sentiment"] = df.apply(
            lambda row: np.nan
            if row[f"{keyword}_Count"] == 0
            else round(
                (
                    row[f"{keyword}_Positive_Count"]
                    - row[f"{keyword}_Negative_Count"]
                    + row[f"{keyword}_Count"]
                )
                / (2 * row[f"{keyword}_Count"]),
                2,
            ),
            axis=1,
        )

    df = df[column_order]
    # saving the dataframe
    logger.info(f"file_name: {file_name}, local_path: {local_path}")
    df.to_csv(local_path, index=False)


def list_new_csv_files(source_dir, start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")

    files = os.listdir(source_dir)
    csv_files = [file for file in files if file.endswith(".csv")]
    filtered_files = []
    for file in csv_files:
        file_date_str = file.split(".")[0]
        parts = file_date_str.split("_")
        file_date = datetime.strptime(f"{parts[0]}_{parts[1]}_{parts[2]}", "%Y_%m_%d")
        if file_date >= start_date:
            filtered_files.append(file)
    return filtered_files


def combine_sentiment_by_callsign(metrics, keyword_list, start_date_str="2024_06_26"):
    """
    Create csv files for each date with data grouped by radio call-sign and save them in "metrics" folder
    """
    source_dir = metrics

    csv_files = list_new_csv_files(source_dir, start_date_str)

    # Define column_order, starting with value ['State', 'Call_Sign']
    column_order = ["State", "Call_Sign"]
    for keyword in keyword_list:
        # Columns per keyword in order
        columns = [
            f"{keyword}_Positive_Count",
            f"{keyword}_Neutral_Count",
            f"{keyword}_Negative_Count",
            f"{keyword}_Count",
            f"{keyword}_Combined_Sentiment",
        ]
        column_order.extend(columns)

    logger.info(f"csv file column_order: {column_order}")

    for csv_file in csv_files:
        file_name = csv_file
        combined_sentiment(source_dir, file_name, column_order, keyword_list)


# Combine data state-wise
def fill_nulls_and_return_mean_columns(source_dir, file_name, keyword_list):

    # Read the CSV file into a DataFrame
    local_path = os.path.join(source_dir, file_name)
    df = pd.read_csv(local_path)

    mean_columns = ["Date", "State"]

    date = file_name.split(".")[0]
    df["Date"] = date

    for keyword in keyword_list:
        # Fill null values in 'Biden_Combined_Sentiment' with the mean value grouped by 'State'
        df[f"{keyword}_Combined_Sentiment"] = df.groupby("State")[
            f"{keyword}_Combined_Sentiment"
        ].transform(lambda x: x.fillna(round(x.mean(), 2)))

        mean_col_name = f"{keyword}_Combined_Sentiment_Mean"
        # Append the mean column name to the list of mean columns
        mean_columns.append(mean_col_name)
        # Compute the mean value grouped by 'State' and store it in a new column
        df[mean_col_name] = (
            df.groupby("State")[f"{keyword}_Combined_Sentiment"]
            .transform("mean")
            .round(4)
        )

    # Return the DataFrame with only 'State' and mean columns
    df = df[mean_columns].drop_duplicates().reset_index(drop=True)
    return df
    # return df


def combine_all_files_in_directory(
    source_dir, dest_dir, keyword_list, start_date_str="2024_06_26"
):
    """
    To combine DataFrames for all files in a directory
    """
    all_files = list_new_csv_files(source_dir, start_date_str)
    combined_df = pd.DataFrame()

    for file_name in all_files:
        df_filled = fill_nulls_and_return_mean_columns(
            source_dir, file_name, keyword_list
        )
        combined_df = pd.concat([combined_df, df_filled], ignore_index=True)

    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, "combined_mean_sentiment_data.csv")
    combined_df.to_csv(local_path, float_format="%.4f", index=False)
    return


if __name__ == "__main__":
    """
    Merge sentiment analysis data and calculate metrics for transcripts starting from date : "start_date_str".
    Default for "start_date_str" is 2024_06_26 in all functions.
    """
    # start_date_str = "2024_06_26"

    base_dir = "assets/analytics/sentiment_analysis"
    log_path = os.path.join(base_dir, "logs/")
    logger.add(f"{log_path}calculate_metrics.log")

    start_time = time.time()
    keyword_list = [
        "Biden",
        "Harris",
        "Trump",
        "Democrats",
        "Republicans",
        "Taylor_Swift",
        "Musk",
        "Olympics",
    ]

    # Merge all json files
    sentiment_buffer = os.path.join(base_dir, "sentiment_buffer")
    merged_json = os.path.join(base_dir, "merged_json")

    merge_json_files(sentiment_buffer, merged_json)
    # merge_json_files(sentiment_buffer, merged_json,  start_date_str)

    # Create temporary csv files with per call sign per transcript data
    metrics = os.path.join(base_dir, "metrics")
    calculate_stats(merged_json, metrics)
    # calculate_stats(merged_json, metrics, start_date_str)

    # Create csv files for each date with data combined by radio call-sign
    combine_sentiment_by_callsign(metrics, keyword_list)
    # combine_sentiment_by_callsign(metrics, keyword_list, start_date_str)

    # Get metrics combined by per state
    final_data_files = os.path.join(base_dir, "final_data_files")
    combine_all_files_in_directory(metrics, final_data_files, keyword_list)
    # combine_all_files_in_directory(metrics, final_data_files, keyword_list, start_date_str)

    logger.info(
        f"State-wise combined mean sentiment data available in {final_data_files} folder"
    )

    time_taken = time.time() - start_time
    logger.info(f"metric calculation done, total time taken: {time_taken:.2f} sec")
