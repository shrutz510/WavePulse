"""
Create listener to check and trigger transcription for any audio files present/pending transcription in audio buffer folders.
- Creates a separate listener for each GPU being used by the application.
- When idle, listeners check for files in audio buffer folders in one minute intervals.
- When audio files found in audio buffer, trigger transcription by call audio_processor.scribe:transcribe_audio function. 
"""

import concurrent.futures
import config
import copy
import os
import time
import threading

from audio_processor.scribe import transcribe_audio
from args import get_args
from functools import partial
from loguru import logger
from typing import List, Dict

# from multiprocessing import Pool


def start_thread_to_terminate_when_parent_process_dies(ppid):
    """
    Stop the backgroung listener when shutting down application.
    """
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def start_multiple_scribe_listener(
    model_parameters: Dict[str, str],
    temp_file_dir: str,
    models_dir: str,
    data_dir: str,
    transcripts_dir: str,
    number_of_gpus: int,
):
    """
    Start multiple listeners that periodically check audio buffer folders for any audio files pending transcription.
    One listener is created per gpu.

    Parameters:
        model_parameters (dict): Dictionary of WhisperX model hyperparameters
        temp_file_dir (str): Audio buffer folder containing audio files pending transcription
        models_dir (str): Directory where Whisper model is downloaded
        data_dir (str): Base data directory for audio and transcript files
        transcripts_dir (str): Directory where transcripts are stored (classified and unclassified)
        number_of_gpus (int): Number of gpus being used for transcription (same number of listeners are created)
    """

    argument_list = []

    for i in range(1, number_of_gpus + 1):
        # creating different model parameters for different GPUs, and assign unique device index for each
        model_parameters_temp = copy.deepcopy(model_parameters)
        model_parameters_temp["device_index"] = i - 1
        arguments = {
            "model_parameters": model_parameters_temp,
            "temp_file_dir": f"{temp_file_dir}_{i}",
        }
        argument_list.append(arguments)

    start_scribe_listener_with_args = partial(
        start_scribe_listener,
        models_dir=models_dir,
        data_dir=data_dir,
        transcripts_dir=transcripts_dir,
    )

    logger.info("starting listener in background ...")
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            len(argument_list),
            initializer=start_thread_to_terminate_when_parent_process_dies,
            initargs=(os.getpid(),),
        )
        executor.map(start_scribe_listener_with_args, argument_list)
        logger.info(f"starting listener in background successfull ...")

    except (KeyboardInterrupt, SystemExit):
        logger.info(f"stopped all {len(argument_list)} listeners")

    return executor


def start_scribe_listener(argument_list, models_dir, data_dir, transcripts_dir) -> None:
    """
    Keep checking argument_list['temp_file_dir'] for .mp3 files, if any found then trigger transcription for it and then delete it.
    Stop the listener if the application is shutting down by checking config.shared_config['running'] periodically

    Parameters:
        argument_list (dict): [
            - temp_file_dir (str): Audio buffer folder to be monitored by this listener.
            - model_parameters (dict): Dictionary containing information about WhisperX model hyperparameters
                - 'batch_size'
                - 'compute_type'
                - 'device'
                - 'whisper_model'
        ]
        models_dir (str): path to ml models directory
        data_dir (str): path to base data directory
        transcripts_dir (str): path to directory that stores transcripts
    """
    model_parameters = argument_list["model_parameters"]
    temp_file_dir = argument_list["temp_file_dir"]
    logger.info("starting scribe listener .....")
    try:
        # Keep running listener in the main thread
        while config.shared_config["running"]:
            logger.info(
                f"checking for audio files pending transcription in {temp_file_dir} config.shared_config['running']: {config.shared_config['running']} ..."
            )
            temp_files = os.listdir(temp_file_dir)
            if temp_files:
                audio_files = []
                for file in temp_files:
                    if file.endswith(".wav") or file.endswith(".mp3"):
                        audio_files.append(os.path.join(temp_file_dir, file))
                logger.info(f"found {len(audio_files)} to be transcribed")

                start_time = time.time()
                number_of_files_transcribed = transcribe_audio(
                    tuple(audio_files), model_parameters, models_dir, transcripts_dir
                )
                end_time = time.time()
                logger.info(
                    f"successfully transcribed {number_of_files_transcribed} files in: {(end_time - start_time):.2f} seconds"
                )

            else:
                logger.info(
                    f"no new audio files found in {temp_file_dir}, sleeping for 60 seconds ..."
                )
                time.sleep(60)
        logger.info(
            f"stopping scribe listener for buffer: {temp_file_dir} as config.shared_config['running']: {config.shared_config['running']}"
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info(f"stopping scribe listener for buffer: {temp_file_dir} .....")

    return


# Driver code
if __name__ == "__main__":
    # To create necessary directories for model and data if not already present
    args = get_args()
    models_dir = os.path.join(args.assets_dir, args.models_dir)
    data_dir = os.path.join(args.assets_dir, args.data_dir)
    temp_file_dir = os.path.join(args.assets_dir, args.data_dir, args.temp_file_dir)
    transcripts_dir = os.path.join(data_dir, args.transcripts_dir)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(temp_file_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    model_parameters = {
        "batch_size": args.whisperx_batch_size,
        "compute_type": args.whisperx_compute_type,
        "device": args.device,
        "whisper_model": args.whisperx_model,
    }

    start_scribe_listener(
        model_parameters, models_dir, data_dir, temp_file_dir, transcripts_dir
    )
