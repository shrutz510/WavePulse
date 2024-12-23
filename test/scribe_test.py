"""
Do speaker diarization to separate out speech signal corresponding to different speakers.
 Transcribe the audio signal and save it as a text file. 
 Do any preprocessing if required in case of multiple speakers.
"""

import os
import unittest

from datetime import date
from src.audio_processor.args import get_args
from src.audio_processor.scribe import transcribe_audio


class TestAudioStreamer(unittest.TestCase):
    def test_transcribe_audio(self):
        args = get_args()
        data_dir = os.path.join(
            args.assets_dir, args.test_data_dir
        )
        models_dir = os.path.join(
            args.assets_dir, args.models_dir
        )
        os.makedirs(data_dir, exist_ok=True)

        model_parameters = {
            "batch_size": args.whisper_batch_size,
            "compute_type": args.whisper_compute_type,
            "device": args.device,
            "whisper_model": args.whisper_model
        }

        string_date = str(date.today()).replace("-", "_")
        audio_files = tuple(
            [
                os.path.join(data_dir, f"audio.wav")
            ]
        )

        string_date = str(date.today()).replace("-", "_")

        expected_files = [
            os.path.join(data_dir, f"audio_transcript.json")
        ]

        # Ensure no leftover files from previous tests
        for file in expected_files:
            if os.path.exists(file):
                os.remove(file)

        # Run the parallel streaming function
        transcribe_audio(audio_files, model_parameters,  models_dir)

        # Verify that the files were created
        for file in expected_files:
            self.assertTrue(os.path.exists(file))
            # Clean up the created files after the test
            if os.path.exists(file):
                os.remove(file)

        # self.assertEqual(output_files, expected_files)
        print("test_transcribe_audio passed.")


if __name__ == "__main__":
    unittest.main()
