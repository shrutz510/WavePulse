"""
Process the audio recording schedule file to set up a background scheduler that streams audio from radio stations at predefined times using cron triggers.
For streams where the scheduled start time has already passed but the end time is still active, dynamically adjust the start time to begin recording one minute from the current moment.
Cron trigger calls the following function to stream audio from multiple radio stations in parallel : audio_processor.audio_streamer:stream_parallel
"""

import json
import os
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from args import get_args
from audio_processor.audio_streamer import stream_parallel
from datetime import datetime, timedelta
from pytz import timezone
from loguru import logger
from typing import List, Dict


def get_duration(start_time: str, end_time: str) -> int:
    """
    Calculate the duration in seconds between two times given in the format "HH:MM".

    Parameters:
        start_time (str): The start time in the format "HH:MM".
        end_time (str): The end time in the format "HH:MM".

    Returns:
        int: The duration in seconds between the start and end times.
    """
    time_format = "%H:%M"
    start = datetime.strptime(start_time, time_format)
    end = datetime.strptime(end_time, time_format)

    # Handle cases where end time is past midnight (i.e., next day)
    if end < start:
        end += timedelta(days=1)

    duration = end - start
    return int(duration.total_seconds())


# def handle_already_started(station_stream_info):
def handle_already_started(station_stream_info: List[Dict]) -> List[Dict]:
    """
    Adjust the start time for streams that have already started but not ended.
    If the stream has already started but the end time has not been reached, set the start time to 2 minute ahead of the current time.

    Parameters:
        station_stream_info (list): A list of dictionaries containing radio streamimg schedule information.
            Each dictionary should have a "time" key, which is a list of tuples with start and end times in the format "HH:MM".

    Returns:
        list: The updated station stream information with adjusted start times.
    """
    now = datetime.now()
    time_format = "%H:%M"
    for station in station_stream_info:
        for i in range(len(station["time"])):
            start_time, end_time = station["time"][i]
            start = datetime.strptime(start_time, time_format)
            start = start.replace(year=now.year, month=now.month, day=now.day)
            end = datetime.strptime(end_time, time_format)
            end = end.replace(year=now.year, month=now.month, day=now.day)

            if end < start:
                try:
                    end = end.replace(year=now.year, month=now.month, day=now.day + 1)
                except ValueError:
                    logger.warning("Rounding up to next month")
                    end = end.replace(year=now.year, month=now.month + 1, day=1)

            # Adjust start time if it has already passed but end time is not within 2 minutes from now
            if start < now and end > now + timedelta(minutes=2):
                start = now + timedelta(minutes=2)
                new_start_time = start.strftime(time_format)
                station["time"][i][0] = new_start_time
    return station_stream_info


def process_schedule_file(radio_schedule_file: str, data_dir: str) -> list[dict]:
    """
    Read radio schedule file and process into usable format for scheduling.

    Parameters:
        radio_schedule_file (str): name of file containing radio schedules (ex: weekly_schedule.json)
    Returns:
        station_schedule_info list(dict): list of dictionary of following elements
            - "time" (str): Time when to start recording in EDT (ex: "00:32"),
            - "radio_list" list(dict): [
                - "url" (str): The URL of the live stream.
                - "radio_name (str)": name of state and stream (ex: "MA_WCBM")
                - "duration" (int): duration to record in seconds
                ]
        write "station_schedule_info" into file processed_schedule.json for easily double checking

    Example:
        >> process_schedules("weekly_schedule.json")
        [
            {
                "time": "00:30",
                "radio_list": [
                    {
                        "url": "https://stream.revma.ihrhls.com/zc3014",
                        "radio_name": "AK_KENI",
                        "duration": 1800
                    },
                    {
                        "url": "http://crystalout.surfernetwork.com:8001/KFNX_MP3",
                        "radio_name": "AZ_KFNX",
                        "duration": 3600
                    }
                ]
            },
            {
                "time": "01:00",
                "radio_list": [
                    {
                        "url": "https://ice8.securenetsystems.net/KAOX",
                        "radio_name": "TX_KAOX",
                        "duration": 2800
                    },
                    {
                        "url": "https://ice10.securenetsystems.net/KINA",
                        "radio_name": "KS_KINA",
                        "duration": 1500
                    }
                ]
            }
        ]
    """
    with open(radio_schedule_file) as f:
        station_stream_info = json.load(f)

    # Handle the cases where start time has already passed but end time has not
    # set their start_time to 2 minutes ahead of current time
    station_stream_info = handle_already_started(station_stream_info)

    # Initialize an empty dictionary to store station schedule info
    station_schedule_info = []

    # Create list (sorting based on start time) to keep track of unique times
    unique_times = []
    for station in station_stream_info:
        for start_time, _ in station["time"]:
            if start_time not in unique_times:
                unique_times.append(start_time)

    unique_times.sort()
    logger.info(f"unique timings: {unique_times}")

    # Iterate over unique timings and group stations accordingly
    for time_slot in unique_times:
        stations_at_time = []
        for station in station_stream_info:
            for start_time, end_time in station["time"]:
                if start_time == time_slot:
                    stations_at_time.append(
                        {
                            "url": station["url"],
                            "radio_name": f"{station['state']}_{station['radio_name']}",
                            "duration": get_duration(start_time, end_time),
                        }
                    )
        station_schedule_info.append(
            {"time": time_slot, "radio_list": stations_at_time}
        )

    # Write station_schedule_info into a file for easy cross verification of schedules
    processed_schedule = os.path.join(data_dir, "processed_schedule.json")
    with open(processed_schedule, "w") as json_file:
        json.dump(station_schedule_info, json_file, indent=4)

    return station_schedule_info


def create_scheduler(
    station_schedule_info: List[Dict],
    segment_duration: int,
    audio_dir: str,
    audio_buffer_dir: str,
    no_of_devices: int,
) -> BackgroundScheduler:
    """
    Creates a background scheduler that records audio streams using
        audio_streamer.py based on schedule from station_schedule_info.
    It is triggered using a CronTrigger and
        calls "audio_processor.audio_streamer:stream_parallel" method
    It passes (time_slot["radio_list"], segment_duration, audio_dir, audio_buffer_dir) as arguments.
    After creating and starting scheduler main process runs in infinite loop till keyboard interrupt.

    Parameters:
        station_schedule_info list(dict): list of dictionary of following elements
            - "time" (str): Time when to start recording in EDT (ex: "00:32"),
            - "radio_list" list(dict): [
                - "url" (str): The URL of the live stream.
                - "radio_name": name of state and stream (ex: "AK_KENI")
                - "duration" (int): duration for which to record stream in seconds
                ]
        segment_duration (int): record audio in segments of this duration (in seconds)
        audio_dir (str): Directory at which audio files are stored
        audio_buffer_dir (str): path to temp audio file directory that need to be transcribed

    Example:
    >> station_schedule_info
    [
        {
            "time": "00:30",
            "radio_list": [
                {
                    "url": "https://stream.revma.ihrhls.com/zc3014",
                    "radio_name": "AK_KENI",
                    "duration": 1800
                }
            ]
        }
    ]
    >> create_scheduler(station_schedule_info, segment_duration, audio_dir, audio_buffer_dir)

    creates a schedule to call method
    src.audio_processor.audio_streamer.stream_parallel(time_slot["radio_list"], segment_duration, audio_dir, audio_buffer_dir)
    where time_slot["radio_list"] = [
                {
                    "url": "https://stream.revma.ihrhls.com/zc3014",
                    "radio_name": "AK_KENI",
                    "duration": 1800
                }
            ]
    """

    # Initialize scheduler
    scheduler = BackgroundScheduler()
    scheduler.configure(timezone=timezone("US/Eastern"))

    # Create trigger/schedules for each time slot
    for time_slot in station_schedule_info:
        hour, min = time_slot["time"].split(":")
        trigger = CronTrigger(hour=hour, minute=min)
        scheduler.add_job(
            "audio_processor.audio_streamer:stream_parallel",
            trigger=trigger,
            args=[
                time_slot["radio_list"],
                segment_duration,
                audio_dir,
                audio_buffer_dir,
                no_of_devices,
            ],
        )

    return scheduler


def create_radio_streaming_scheduler(
    data_dir: str,
    audio_dir: str,
    audio_buffer_dir: str,
    radio_schedule_file: str,
    segment_duration: int,
    no_of_devices: int,
) -> BackgroundScheduler:
    """
    Process radio schedules from schedule.json file in assets directory.
    Create and return background scheduler for recording audio streams.
    Parameters:
        data_dir (str): base directory for all data like audio and transcript files
        audio_dir (str): directory for saving recorded audio files
        audio_buffer_dir (str): buffer directory for storing audio files pending transcription
        radio_schedule_file (str): json file with schedule for streaming radio stations
        segment_duration (int): streamed audio to be recorded in segments of this duration
        no_of_devices (int): number of gpus being used for transcription
    """
    station_schedule_info = process_schedule_file(radio_schedule_file, data_dir)

    return create_scheduler(
        station_schedule_info,
        segment_duration,
        audio_dir,
        audio_buffer_dir,
        no_of_devices,
    )


if __name__ == "__main__":

    # To create necessary directories for model and data if not already present
    args = get_args()
    data_dir = os.path.join(args.assets_dir, args.data_dir)
    audio_dir = os.path.join(args.assets_dir, args.data_dir, args.audio_dir)
    audio_buffer_dir = os.path.join(
        args.assets_dir, args.data_dir, args.audio_buffer_dir
    )
    radio_schedule_file = os.path.join(args.assets_dir, args.radio_schedule)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(audio_buffer_dir, exist_ok=True)
    logger.info(f"all the required directories for audio streaming created/exist")
    segment_duration = args.segment_duration

    scheduler = create_radio_streaming_scheduler(
        data_dir, audio_dir, audio_buffer_dir, radio_schedule_file, segment_duration
    )

    logger.info(f"starting radio scheduler .....")
    scheduler.start()
    try:
        # This is here to keep the main thread alive
        while True:
            logger.info(f"sleeping for 30 seconds zzz.......")
            time.sleep(30)

    except (KeyboardInterrupt, SystemExit):
        scheduler.remove_all_jobs()
        scheduler.shutdown()
        logger.info("stopped radio scheduler .....")
