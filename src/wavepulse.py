"""
Controller for audio_processor and fact_checker modules.
 Trigger audio_processor at set timmings for different radio stations.
 Process the radio transcript using matcher to find mis-information and compute metrics.

 $ python -m src.observatory
"""

import config
import os
import time
import torch
from args import get_args
from audio_processor.radio_scheduler import create_radio_streaming_scheduler
from fact_checker.factcheck_scheduler import create_factcheck_scheduler
from audio_processor.scribe_listener import start_multiple_scribe_listener
from text_processor.classification_listener import \
    start_classification_listener
from datetime import datetime
from loguru import logger
from utils.backup import backup_files_via_ftp

# from utils.apscheduler_shutdown import shutdown_scheduler_and_processes

# Setup logging
args = get_args()
log_dir = os.path.join(args.assets_dir, args.logs_dir, "app.log")
logger.add(log_dir, format="{time} {level} {message}", level="INFO")


def stop_background_processes(
        radio_scheduler, factcheck_scheduler, executor, classification_executor
):
    logger.info("application shuting down ...")
    config.shared_config["running"] = False
    if radio_scheduler:
        radio_scheduler.remove_all_jobs()
        radio_scheduler.shutdown(wait=False)
    if factcheck_scheduler:
        factcheck_scheduler.remove_all_jobs()
        factcheck_scheduler.shutdown(wait=False)
    if executor:
        executor.shutdown(wait=False, cancel_futures=True)
    if classification_executor:
        classification_executor.shutdown(wait=False, cancel_futures=True)
    return


def start_observatory():
    """
    Main driver code to initialize application.
    Create and start scheduler to automatically record radio streams.
    Start scribe_listener to keep checking if any new audio files need to be transcribed.
    """
    # TODO dont hard code. do this by default but give control for which all devices to use
    #   e.g., 0,2 will only use first and third GPUs.
    number_of_gpus = torch.cuda.device_count()

    # To create necessary directories for model and data if not already present
    args = get_args()
    models_dir = os.path.join(args.assets_dir, args.models_dir)
    data_dir = os.path.join(args.assets_dir, args.data_dir)
    audio_dir = os.path.join(args.assets_dir, args.data_dir,
                             args.recordings_dir)
    transcripts_dir = os.path.join(args.assets_dir, args.data_dir,
                                   args.transcripts_dir)
    audio_buffer_dir = os.path.join(
        args.assets_dir, args.data_dir, args.audio_buffer_dir
    )
    factcheck_dir = os.path.join(args.assets_dir, args.data_dir,
                                 args.factcheck_dir)
    radio_schedule_file = os.path.join(args.assets_dir, args.radio_schedule)

    transcripts_temp_dir = os.path.join(transcripts_dir, "unclassified_buffer")
    classified_file_dir = os.path.join(transcripts_dir, "classified")
    classified_json_file_dir = os.path.join(classified_file_dir, "json")
    classified_political_file_dir = os.path.join(classified_file_dir,
                                                 "political")
    classified_political_ad_file_dir = os.path.join(classified_file_dir,
                                                    "political_ad")
    classified_apolitical_file_dir = os.path.join(classified_file_dir,
                                                  "apolitical")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    # os.makedirs(audio_buffer_dir, exist_ok=True)
    os.path.isfile(radio_schedule_file)
    for i in range(number_of_gpus):
        os.makedirs(f"{audio_buffer_dir}_{i + 1}", exist_ok=True)
        tmp = f"{audio_buffer_dir}_{i + 1}"
        n = len(os.listdir(f"{audio_buffer_dir}_{i + 1}"))
        logger.info(f"number of files in {tmp} :{n}")

    os.makedirs(transcripts_temp_dir, exist_ok=True)
    os.makedirs(classified_file_dir, exist_ok=True)
    os.makedirs(classified_json_file_dir, exist_ok=True)
    os.makedirs(classified_political_file_dir, exist_ok=True)
    os.makedirs(classified_political_ad_file_dir, exist_ok=True)
    os.makedirs(classified_apolitical_file_dir, exist_ok=True)

    logger.info(f"all the required directories created/exist")

    segment_duration = args.segment_duration
    start_date = args.start_date
    end_date = args.end_date
    start_page = args.start_page
    end_page = args.end_page
    title_keys = args.title_keys
    tags = args.tags
    political_keywords = args.political_keywords
    spiders = args.spiders

    scriber_model_parameters = {
        "batch_size": args.whisperx_batch_size,
        "compute_type": args.whisperx_compute_type,
        "device": args.device,
        "whisper_model": args.whisperx_model,
    }

    logger.info(f"whisper model parameters: {scriber_model_parameters}")

    executor = None
    factcheck_scheduler = None

    start_time = time.time()

    repetitions = args.no_of_repetition
    shutdown_time = datetime.strptime(args.shutdown_time, "%H:%M").time()
    restart_time = datetime.strptime(args.restart_time, "%H:%M").time()
    restart_flag = 1
    try:
        for i in range(repetitions):
            flag = 1
            config.shared_config["running"] = True
            logger.info(
                f"config.shared_config['running'] : {config.shared_config['running']}"
            )
            # start radio scheduler for streaming audio and saving it in audio dir
            if not args.stop_recording:
                # Create and start radio streaming scheduler
                logger.info(
                    f"creating and starting radio streaming scheduler .....")
                radio_scheduler = create_radio_streaming_scheduler(
                    data_dir,
                    audio_dir,
                    audio_buffer_dir,
                    radio_schedule_file,
                    segment_duration,
                    number_of_gpus,
                )
                radio_scheduler.start()
                logger.info(f"radio scheduler running .....")

            else:
                logger.info(f"audio recording off ....")
                radio_scheduler = None

            # Create and start listener in background to transcribe audio files (.mp3 and .wav) in /assets/data/temp folder
            if not args.stop_transcription:
                executor = start_multiple_scribe_listener(
                    scriber_model_parameters,
                    audio_buffer_dir,
                    models_dir,
                    data_dir,
                    transcripts_dir,
                    number_of_gpus,
                )
                logger.info(f"scribe listener running .....")
            else:
                logger.info("transcription off ....")
                executor = None

            # Classification of text of transcript into Political - Apolitical Content and Ad - Not Ad
            if not args.stop_classification:
                classification_executor = start_classification_listener(
                    transcripts_dir)
                logger.info(f"classification listener running .....")
            else:
                logger.info("classification off ....")
                classification_executor = None

            # Create and start fact check scheduler
            logger.info(f"Creating fact check scheduler ...")
            factcheck_scheduler = create_factcheck_scheduler(
                factcheck_dir,
                start_date,
                end_date,
                start_page,
                end_page,
                title_keys,
                tags,
                political_keywords,
                spiders,
            )
            logger.info(f"Starting fact check scheduler ...")
            factcheck_scheduler.start()
            logger.info(f"Fact check scheduler running ...")

            while flag:
                time.sleep(10)
                if datetime.now().time() > shutdown_time \
                        and datetime.now().time() < restart_time \
                        and restart_flag == 1:
                    if i < repetitions - 1:
                        logger.info(
                            f"stopping radio scheduler and scribe listeners")
                        restart_flag = 0
                        stop_background_processes(
                            radio_scheduler,
                            factcheck_scheduler,
                            executor,
                            classification_executor,
                        )
                        time.sleep(90)
                        if args.backup_audio:
                            backup_files_via_ftp(audio_dir)
                    else:
                        logger.info("final application shutdown started")
                        stop_background_processes(
                            radio_scheduler,
                            factcheck_scheduler,
                            executor,
                            classification_executor,
                        )
                        time.sleep(90)
                        if args.backup_audio:
                            backup_files_via_ftp(audio_dir)
                        # Set flag to 0 get out of infinite while loop
                        flag = 0
                if datetime.now().time() >= restart_time and restart_flag == 0:
                    logger.info(
                        f"restarting radio scheduler and scribe listeners")
                    # Set restart_flag to 1 so that app can be shutdown again when time comes
                    restart_flag = 1
                    # Set flag to 0 get out of infinite while loop
                    flag = 0

    except (KeyboardInterrupt, SystemExit):
        logger.info(f"shutting down observatory")
        stop_background_processes(
            radio_scheduler, factcheck_scheduler, executor,
            classification_executor
        )

    logger.info("stopped radio scheduler.")
    end_time = time.time()
    logger.info(f"total time application ran: {(end_time - start_time):.2f}s")


if __name__ == "__main__":
    start_observatory()
