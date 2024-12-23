"""
Stream audio from multiple radio stations at a time and save it in an audio file to be processed.
To run the test execute from root directory: 
  >>> python -m test.audio_processor.audio_streamer_test

"""

import os
import unittest

# from unittest.mock import patch, MagicMock

from datetime import date
from src.audio_processor.audio_streamer import run_parallel
from src.audio_processor.args import get_args


class TestAudioStreamer(unittest.TestCase):
    def test_run_parallel(self):
        # station_stream_list = [
        #     {'url': 'https://crystalout.surfernetwork.com:8001/KGWA_MP3', 'radio_name': 'KGWA'},
        #     {'url': 'http://stream.revma.ihrhls.com/zc5225', 'radio_name': 'WAAX'}
        # ]
        args = get_args()
        data_dir = os.path.join(
            args.assets_dir, args.test_data_dir
        )
        os.makedirs(data_dir, exist_ok=True)

        station_stream_list = [
            {
                "url": "https://crystalout.surfernetwork.com:8001/KGWA_MP3",
                "radio_name": "KGWA",
            },
            {"url": "http://stream.revma.ihrhls.com/zc5225", "radio_name": "WAAX"},
            {"url": "https://stream.revma.ihrhls.com/zc3014", "radio_name": "KENI"},
            {
                "url": "http://crystalout.surfernetwork.com:8001/KFNX_MP3",
                "radio_name": "KFNX",
            },
            {"url": "http://ophanim.net:9760/stream", "radio_name": "KURM"},
            {"url": "http://stream.revma.ihrhls.com/zc401", "radio_name": "KCOL"},
            {"url": "https://ice41.securenetsystems.net/WHBO", "radio_name": "WHBO"},
            {"url": "https://ice8.securenetsystems.net/KAOX", "radio_name": "KAOX"},
            {"url": "http://ice41.securenetsystems.net/WBGZ", "radio_name": "WBGZ"},
            {
                "url": "http://crystalout.surfernetwork.com:8001/KBIZ_MP3",
                "radio_name": "KBIZ",
            },
            {"url": "https://ice10.securenetsystems.net/KINA", "radio_name": "KINA"},
            {"url": "https://us2.maindigitalstream.com/ssl/WHIR", "radio_name": "WHIR"},
            {"url": "https://ice9.securenetsystems.net/KFXZAM", "radio_name": "KFXZ"},
            # {"url": "http://ic1.mainstreamnetwork.com/wlob-am", "radio_name": "WLOB"},    audio stream stopped working ??
            {
                "url": "https://14153.live.streamtheworld.com/WCBMAM_SC",
                "radio_name": "WCBM",
            },
            {"url": "http://stream.revma.ihrhls.com/zc7729", "radio_name": "WBZ"},
        ]
        n = len(station_stream_list)

        # To repeat audio streams to test max possible streams that can be handled by the system in parallel
        repetitions = 0
        for j in range(2**repetitions - 1):
            print(f"repetiotion j:{j}")
            for i in range(n):
                station_stream_list.append(
                    {
                        "url": station_stream_list[i]["url"],
                        "radio_name": station_stream_list[i]["radio_name"]
                        + "_"
                        + str(j),
                    }
                )

        string_date = str(date.today()).replace("-", "_")

        expected_files = tuple(
            [
                os.path.join(data_dir, f"KGWA_{string_date}.wav"),
                os.path.join(data_dir, f"WAAX_{string_date}.wav"),
                os.path.join(data_dir, f"KENI_{string_date}.wav"),
                os.path.join(data_dir, f"KFNX_{string_date}.wav"),
                os.path.join(data_dir, f"KURM_{string_date}.wav"),
                os.path.join(data_dir, f"KCOL_{string_date}.wav"),
                os.path.join(data_dir, f"WHBO_{string_date}.wav"),
                os.path.join(data_dir, f"KAOX_{string_date}.wav"),
                os.path.join(data_dir, f"WBGZ_{string_date}.wav"),
                os.path.join(data_dir, f"KBIZ_{string_date}.wav"),
                os.path.join(data_dir, f"KINA_{string_date}.wav"),
                os.path.join(data_dir, f"WHIR_{string_date}.wav"),
                os.path.join(data_dir, f"KFXZ_{string_date}.wav"),
                # os.path.join(data_dir, f"WLOB_{string_date}.wav"),    audio stream stopped working ??
                os.path.join(data_dir, f"WCBM_{string_date}.wav"),
                os.path.join(data_dir, f"WBZ_{string_date}.wav"),
            ]
        )

        # Ensure no leftover files from previous tests
        for file in expected_files:
            if os.path.exists(file):
                os.remove(file)

        # Run the parallel streaming function
        output_files = run_parallel(station_stream_list, 2, data_dir)

        # Verify that the files were created
        for file in expected_files:
            self.assertTrue(os.path.exists(file))
            # Clean up the created files after the test
            if os.path.exists(file):
                os.remove(file)

        print(expected_files)
        self.assertEqual(output_files, expected_files)
        print("test_run_parallel_with_real_urls passed.")


if __name__ == "__main__":
    unittest.main()
