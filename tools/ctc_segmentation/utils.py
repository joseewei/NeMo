# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from pathlib import PosixPath
from typing import List, Union

import ctc_segmentation as cs
import numpy as np

from nemo.utils import logging

__all__ = ['convert_mp3_to_wav']

MAX_PROB = -10000000000.0
FRAME_DURATION_IN_MS = 40


def get_segments(
    log_probs: np.ndarray,
    path_wav: Union[PosixPath, str],
    transcript_file: Union[PosixPath, str],
    output_file: str,
    vocabulary: List[str],
    window_size: int = 8000,
    frame_duration_ms: int = 20,
) -> None:
    """
    Segments the audio into segments and saves segments timings to a file

    Args:
        log_probs: Log probabilities for the original audio from an ASR model, shape T * |vocabulary|.
                   values for blank should be at position 0
        path_wav: path to the audio .wav file
        transcript_file: path to
        output_file: path to the file to save timings for segments
        vocabulary: vocabulary used to train the ASR model, note blank is at position 0
        window_size: the length of each utterance (in terms of frames of the CTC outputs) fits into that window. The default window is 8000, your audio file is much shorter. You may reduce this value to improve alignment speed.
        frame_duration_ms: frame duration in ms
    """
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.min_window_size = window_size
    config.frame_duration_ms = frame_duration_ms
    config.blank = config.space
    config.subsampling_factor = 2

    with open(transcript_file, "r") as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    logging.info(f"Syncing {transcript_file}")
    ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)

    logging.info(
        f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
    )

    if len(ground_truth_mat) > log_probs.shape[0]:
        logging.warning(f"Skipping: Audio {path_wav} is shorter than text {transcript_file}")
    else:
        start_time = time.time()
        timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
        total_time = time.time() - start_time
        logging.info(f"Time: {total_time}s ---> ~{round(total_time/60)}min")
        logging.info(f"Saving segments to {output_file}")

        segments = cs.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
        write_output(output_file, path_wav, segments, text)


def write_output(out_path, path_wav, segments, text, stride: int = 1):
    """

    :param out_path:
    :param path_wav:
    :param segments:
    :param text:
    :param stride: Stride applied to an ASR input, for example, for QN 10 ms after stride is applied 20 ms
    :return:
    """
    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        for i, (start, end, score) in enumerate(segments):
            outfile.write(f'{start/stride} {end/stride} {score} | {text[i]}\n')


def convert_mp3_to_wav(mp3_file: str, wav_file: str = None, sampling_rate: int = 16000) -> str:
    """
    Converts .mp3 to .wav and changes sampling rate if needed

    mp3_file: Path to .mp3 file
    sampling_rate: Desired sampling rate

    Returns:
        path to .wav file
    """
    logging.info(f"Converting {mp3_file} to .wav format with sampling rate {sampling_rate}")

    if wav_file is None:
        wav_file = mp3_file.replace(".mp3", ".wav")
    os.system(f'ffmpeg -i {mp3_file} -ac 1 -af aresample=resampler=soxr -ar {sampling_rate} {wav_file} -y')
    return wav_file
