"""
Stream audio from multiple radio stations concurrently, saving the recordings in 30-minute segments (configurable) for easier downstream processing.
- Each recorded audio file is saved in a dedicated recordings folder for backup purposes. A temporary copy is placed in the audio_buffer folder for transcription.
- Temporary files in the audio_buffer folder are deleted automatically after transcription is completed.
- If the application is running on multiple GPUs, a separate audio_buffer folder is created for each GPU. Audio files are distributed across these folders using load balancing to optimize transcription performance.
"""

import config
import m3u8
import os
import requests
import shutil
import subprocess
import time

from datetime import datetime
from functools import partial
from loguru import logger
from multiprocessing import Pool
from pathlib import Path
from pydub import AudioSegment
from requests.exceptions import RequestException
from typing import List, Dict, Optional
from urllib.parse import urljoin


def record_segment(
    url: str, output_file: str, segment_duration: int, retries: int, wait_time: int
) -> bool:
    """
    Records a 30 min (segment_duration) segment of a live stream.

    Parameters:
        url (str): The URL of the live stream.
        output_file (str): The file path to save the recorded segment.
        segment_duration (int): The duration (in seconds) of the segment.
        retries (int): Number of retry attempts for streaming audio.
        wait_time (int): Wait time between consecutive retry attempts.

    Returns:
        bool: True if the recording was successful, False otherwise.
    """
    retries = retries + 1
    start_time = time.time()
    end_time = start_time + segment_duration

    for attempt in range(retries):
        try:
            logger.info(
                f"Attempting to record {url} to {output_file}, attempt {attempt + 1}"
            )
            if "m3u8" in url:
                with open(output_file, "wb") as f:
                    while time.time() < end_time:
                        playlist = m3u8.load(url)
                        if not playlist.segments:
                            logger.error("No segments found in the playlist")
                            return False

                        # Process each segment
                        for segment in playlist.segments:
                            if time.time() >= end_time:
                                break

                            # Construct absolute URL for the segment
                            segment_url = urljoin(url, segment.uri)
                            # logger.info(f"Downloading segment: {segment_url}")

                            # Download and write the segment
                            response = requests.get(segment_url, timeout=10)
                            f.write(response.content)

                            # Wait for the playlist refresh interval
                            time.sleep(playlist.target_duration or 5)

                            if not config.shared_config["running"]:
                                logger.info(
                                    f"stopped recording as application is shutting down"
                                )
                                return False

            else:
                response = requests.get(url, stream=True, timeout=10)
                with open(output_file, "wb") as f:
                    start_time = time.time()
                    while time.time() - start_time < segment_duration:
                        chunk = response.iter_content(chunk_size=512)
                        if chunk:
                            f.write(next(chunk))
                        if not config.shared_config["running"]:
                            logger.info(
                                f"stopped recording as application is shutting down"
                            )
                            return False
                logger.info(f"Successfully recorded {url} to {output_file}")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred for {url}, retrying...")

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error for {url}, retrying...")

        except RequestException as e:
            logger.error(f"Failed to record {url}: {e}")
            break  # For other request exceptions, break the loop
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"stopping recording {url} to shut down app")
            return False

        time.sleep(wait_time)

    else:
        logger.error(f"Failed to record {url} after {retries} attempts")
        return False


def copy_to_buffer(
    audio_buffer_dir: str, file_name: str, output_file: str, no_of_devices: int
) -> None:
    # Load balancing buffer directory
    temp_folder = 1
    min_files = len(os.listdir(f"{audio_buffer_dir}_{1}"))
    for i in range(2, no_of_devices + 1):
        curr_files = len(os.listdir(f"{audio_buffer_dir}_{i}"))
        if min_files > curr_files:
            min_files = curr_files
            temp_folder = i

    curr_audio_buffer_dir = f"{audio_buffer_dir}_{temp_folder}"
    temp_output_file = os.path.join(curr_audio_buffer_dir, file_name)

    shutil.copy(output_file, temp_output_file)
    logger.info(
        f"Successfully copied file {file_name} to temp folder: {curr_audio_buffer_dir}"
    )
    return


def record_live_stream(
    station_info: Dict[str, str], retries: int, wait_time: int, no_of_devices: int
) -> Optional[str]:
    """
    Records a live stream from a given URL and saves it as multiple MP3 files in 5-minute batches.

    Parameters:
    station_info (dict): A dictionary containing the following keys:
        - 'url' (str): The URL of the live stream.
        - 'radio_name' (str): Name of radio being streamed in format State abbreviation followed by radio callsign
        - 'audio_dir' (str): Directory for saving recorded audio files.
        - 'audio_buffer_dir' (str): Buffer directory for storing audio files pending transcription.
        - 'duration' (int): The total duration (in seconds) for which to record the stream.
        - 'segment_duration' (int): Record audio in segments of this duration (in seconds)

    Returns:
     The last output_file name if successful else None

    Example:
        >> station_info = {
                'url': 'https://example.com/stream',
                'radio_name': 'NY_ABCD'
                'audio_dir': 'recordings',
                'audio_buffer_dir': 'audio_buffer',
                'duration': 3600,
                'segment_duration': 1800
            }
        >> record_live_stream(station_info)
        data/audio/recordings/NY_ABCD_2024_01_01_08_00.wav
        data/audio/recordings/NY_ABCD_2024_01_01_08_30.wav
        data/audio_buffer_1/NY_ABCD_2024_01_01_08_00.wav
        data/audio_buffer_1/NY_ABCD_2024_01_01_08_30.wav

    Notes: It will stream and record audio in 30-minute chunks and store them in files like NY_ABCD_2024_01_01_02_06.wav
    """

    url, radio_name, total_duration, audio_dir, audio_buffer_dir, segment_duration = (
        station_info[key]
        for key in (
            "url",
            "radio_name",
            "duration",
            "audio_dir",
            "audio_buffer_dir",
            "segment_duration",
        )
    )

    # Calculate the number of 30-minute segments
    num_segments = total_duration // segment_duration
    remaining_duration = total_duration % segment_duration

    for segment in range(num_segments):
        current_time = datetime.now()
        string_datetime = current_time.strftime("%Y_%m_%d_%H_%M")
        file_name = f"{radio_name}_{string_datetime}.mp3"

        output_file = os.path.join(audio_dir, file_name)

        if not record_segment(url, output_file, segment_duration, retries, wait_time):
            return None  # Return None if segment recording failed

        copy_to_buffer(audio_buffer_dir, file_name, output_file, no_of_devices)

        # Sleep until the next 5-minute segment starts
        time_to_sleep = segment_duration - (time.time() - current_time.timestamp())
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    # Handle the last segment if remaining duration is not zero
    if remaining_duration > 0:
        current_time = datetime.now()
        string_datetime = current_time.strftime("%Y_%m_%d_%H_%M")
        file_name = f"{radio_name}_{string_datetime}.mp3"

        output_file = os.path.join(audio_dir, file_name)

        if not record_segment(url, output_file, remaining_duration, retries, wait_time):
            return None  # Return None if segment recording failed

        # Copy the recorded file to temp_output_file to be transcribed
        copy_to_buffer(audio_buffer_dir, file_name, output_file, no_of_devices)

    logger.info(f"Exiting record_live_stream, last file: {output_file}")
    return output_file


def stream_parallel(
    station_stream_list: List[Dict[str, str]],
    segment_duration: int = 1800,
    audio_dir: str = "assets/data/audio",
    audio_buffer_dir: str = "assets/data/audio_buffer",
    no_of_devices: int = 1,
    retries: int = 5,
    wait_time: int = 60,
) -> None:
    """
    Records multiple live streams in parallel onto disk.
    Sets name of output file based on station name and current date.

    Parameters:
        station_stream_list (list of dict): A list of dictionaries, each containing:
            - 'url' (str): The URL of the live stream.
            - 'radio_name' (str): The name of the radio station.
            - 'duration' (int): duration for which to record stream (in seconds)
        segment_duration (int): record audio in segments of this duration (in seconds)
        audio_dir (str): Directory at which streamed audio files will be stored.
        audio_buffer_dir (str): Directory at which copied audio files pending transcription will be stored
        no_of_devices (int): number of gpus being used for transcription
        retries (int): number of retries to try to record audio
        wait_time (int): wait time between consecutive retry attempts (in seconds)

    Returns:
     None

    Example:
        >> station_stream_list = {
                {'url': 'http://example.com/stream', 'radio_name': 'ST_EXMP', 'duration':1800},
                {'url': 'http://test.com/stream', 'radio_name': 'ST_TEST', 'duration':3000}
                }
        >> stream_parallel(station_stream_list, 1800, "assets/data")
        following files will be created :
        [
            "assets/data/audio/ST_EXMP_2024_01_01_08_00.wav",
            "assets/data/audio/ST_TEST_2024_01_01_08_00.wav",
            "assets/data/audio/ST_TEST_2024_01_01_08_30.wav",

            "assets/data/temp/ST_EXMP_2024_01_01_08_00.wav",
            "assets/data/temp/ST_TEST_2024_01_01_08_00.wav",
            "assets/data/temp/ST_TEST_2024_01_01_08_30.wav"
        ]
        ["assets/data/ST_EXMP", "assets/data/ST_TEST"]
    """

    logger.info(f"storing audio files in directory: {audio_dir}")
    logger.info(f"buffer directory base name: {audio_buffer_dir}")

    station_info_list = []

    # Iterate through station_stream_list to create station_info_list and output_files
    for station in station_stream_list:
        station_info = {
            "url": station["url"],
            "radio_name": station["radio_name"],
            "audio_dir": audio_dir,
            "audio_buffer_dir": audio_buffer_dir,
            "duration": station["duration"],
            "segment_duration": segment_duration,
        }
        station_info_list.append(station_info)

    logger.info(station_info_list)

    record_live_stream_with_args = partial(
        record_live_stream,
        retries=retries,
        wait_time=wait_time,
        no_of_devices=no_of_devices,
    )
    # Use multiprocessing Pool to run record_live_stream function in parallel
    with Pool(processes=len(station_info_list)) as pool:
        result_files = pool.map(record_live_stream_with_args, station_info_list)

    output_files = tuple(file for file in result_files if file is not None)
    logger.debug(output_files)

    logger.info(f"Successfully recorded for {len(output_files)} stations")

    return


# Driver code
if __name__ == "__main__":
    # from args import get_args

    # args = get_args()
    data_dir = "data"  # str(os.path.join(args.assets_dir, args.data_dir))
    recordings_dir = (
        "recordings"  # os.path.join(args.assets_dir, args.data_dir, "recordings")
    )
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(recordings_dir, exist_ok=True)

    station_stream_list = [
        {
                "url": "http://stream.revma.ihrhls.com/zc5225",
                "radio_name": "AL_WAAX",
                "duration": 1800,
            },
    ]

    # Stream and record audio radio stations in station_stream_list
    logger.info(f"number of radio stations: {len(station_stream_list)}")

    start_time = time.time()
    segment_duration = 1800  # args.segment_duration
    stream_parallel(
        station_stream_list, segment_duration, data_dir, retries=5, wait_time=60
    )
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time}")
