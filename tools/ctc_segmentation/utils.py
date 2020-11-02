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

import logging
import logging.handlers
import multiprocessing
import time
from pathlib import PosixPath
from typing import List, Union

import ctc_segmentation as cs
import numpy as np

__all__ = ['get_segments']


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

    # logging.debug(f"Syncing {transcript_file}")
    ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)

    # logging.debug(
    #     f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
    # )

    start_time = time.time()
    timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
    total_time = time.time() - start_time
    logging.info(f"Time: ~{round(total_time/60)}min. Saving segments to {output_file}")

    segments = cs.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
    write_output(output_file, path_wav, segments, text)


def write_output(out_path, path_wav, segments, text, stride: int = 2, offset: float = 0.18):
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
            outfile.write(f'{start/stride} {end/stride + offset} {score} | {text[i]}\n')


#####################
# logging utils
#####################


def listener_configurer(log_file, level):
    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler(log_file, 'a')
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    ch = logging.StreamHandler()
    root.addHandler(h)
    root.setLevel(level)
    root.addHandler(ch)


def listener_process(queue, configurer, log_file, level):
    configurer(log_file, level)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.setLevel(logging.INFO)
            logger.handle(record)  # No level or filter logic applied - just do it!

        except Exception:
            import sys, traceback

            print('Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue, level):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(level)


def worker_process(
    queue, configurer, level, log_probs, path_wav, transcript_file, output_file, vocabulary, window_len
):
    configurer(queue, level)
    name = multiprocessing.current_process().name
    import random

    innerlogger = logging.getLogger('worker')

    innerlogger.info(f'{name} is processing {path_wav} - {random.randint(0, 100)}')
    get_segments(log_probs, path_wav, transcript_file, output_file, vocabulary, window_len)
