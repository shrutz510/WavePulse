"""
Transcribe the audio file in audio buffer folder and save it in json format. Save the transcription in unclassified_buffer folder for it's content to be classified.
- If speaker diarization is on then do it to separate out speech signal corresponding to different speakers.
- Delete the temporary audio file in audio buffer after transcription is complete.
"""

import config
import contextlib
import gc
import json
import os
import time
import torch
import whisperx

from args import get_args
from datetime import datetime, timedelta
from loguru import logger
from typing import Tuple, Dict
from utils.timezone_converter import convert_timezone


def transcribe_audio(
    audio_files: Tuple[str],
    model_parameters: Dict[str, str],
    models_dir: str = "assets/models",
    transcripts_dir: str = "assets/data/transcripts",
) -> int:
    """
    Transcribe audio files using WhisperX and save them.

    Parameters:
        - audio_files (tuple): A tuple of audio file names that need to be transcribed.
        - model_parameters (dict): [
            - 'device' (str): device to load WhisperX model (cpu or cuda)
            - 'device_index' (int): index of gpu on which to load WhisperX model
            - 'batch_size' (int): WhisperX batch size
            - 'compute_type' (str): WhisperX compute type ex. float16, float32
            - 'whisper_model' (str): Whisper model to use ex. small, medium, large-v3 etc
        ]
        - models_dir (str): Directory at which WhisperX model will be downloaded.
        - transcripts_dir (str): Path to directory that stores transcripts
    Returns:
        int: Number of files transcribed in current batch
    Example:
        >>> audio_files = [audio_1.mp3, audio_2.mp3]
        >>> transcribe_audio(audio_files, model_parameters)

        Generate and save files : audio_1_transcript.json and audio_2_transcript.json
    """

    logger.info(f"transcribing for {len(audio_files)} audio files")

    args = get_args()
    diarize = args.diarize
    hf_token = args.hf_token

    # Setup model parameters and load model
    device = model_parameters["device"]
    device_index = model_parameters["device_index"]
    batch_size = model_parameters["batch_size"]
    compute_type = model_parameters["compute_type"]
    whisper_model = model_parameters["whisper_model"]

    asr_options = {
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    logger.debug(f"Loading model with device index: {device_index}")

    model = whisperx.load_model(
        whisper_model,
        device,
        device_index,
        compute_type=compute_type,
        language="en",
        download_root=models_dir,
        asr_options=asr_options,
    )

    if diarize:
        torch_device = f"{device}:{device_index}"
        model_a, metadata = whisperx.load_align_model("en", device=torch_device)
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=torch_device
        )

    logger.info("Loaded models")

    number_of_files_transcribed = 0

    for audio_file in audio_files:
        if not config.shared_config["running"]:
            break
        try:
            # Load audio file
            audio = whisperx.load_audio(audio_file)
            logger.info(f"starting transcription for {audio_file} ....")
            start_time = time.time()
            # 1. Transcribe audio
            result = model.transcribe(audio, batch_size=batch_size, language="en")
            if diarize:
                # 2. Align whisper output
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    torch_device,
                    return_char_alignments=False,
                )
                # 3. Assign speaker labels
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                for segment in result["segments"]:
                    del segment["words"]

            end_time = time.time()
            logger.info(f"Transcription completed in: {(end_time - start_time):.2f}s")

            # Save result to a JSON file
            audio_file_name = os.path.basename(audio_file)
            output_json_file = os.path.join(
                transcripts_dir, "unclassified_buffer", f"{audio_file_name[:-4]}.json"
            )

            with open(output_json_file, "w") as json_file:
                json.dump(result["segments"], json_file, indent=4)

            # output_txt_file = os.path.join(transcripts_dir,
            #                            f"{audio_file_name[:-4]}.txt")
            # reformat_and_save(audio_file_name, output_txt_file, result, diarize)

            number_of_files_transcribed += 1
            del audio
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error transcribing {audio_file}: {e}", exc_info=True)

        # Deleting the audio file from temp folder after transcription is complete
        with contextlib.suppress(FileNotFoundError):
            os.remove(audio_file)

    del model
    if diarize:
        del model_a
        del diarize_model
    gc.collect()
    torch.cuda.empty_cache()

    return number_of_files_transcribed


# Driver code
if __name__ == "__main__":
    args = get_args()
    models_dir = os.path.join(args.assets_dir, args.models_dir)
    data_dir = os.path.join(args.assets_dir, args.data_dir)
    transcripts_dir = os.path.join(args.assets_dir, args.data_dir, args.transcripts_dir)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    logger.info(f"all the required directories created/exist")

    audio_files_list = []
    files = ["KAOX_2024_06_13_0.wav"]

    for file in files:
        audio_file = os.path.join(
            args.assets_dir, args.data_dir, args.temp_file_dir, file
        )
        audio_files_list.append(audio_file)

    model_parameters = {
        "batch_size": args.whisper_batch_size,
        "compute_type": args.whisper_compute_type,
        "device": args.device,
        "whisper_model": args.whisper_model,
    }

    logger.info(f"whisper model parameters: {model_parameters}")

    audio_files = tuple(audio_files_list)
    transcribe_audio(audio_files, model_parameters, models_dir, transcripts_dir)
