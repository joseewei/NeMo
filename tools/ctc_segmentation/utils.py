# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
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

# This file contains code artifacts adapted from the original implementation:
# https://github.com/cornerfarmer/ctc_segmentation

# isort:skip_file

import os
import pickle
import time

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, build_dir="build", build_in_temp=False)
from align_fill_cython import cython_fill_table

from nemo.utils import logging
from pathlib import PosixPath
from typing import Union, List


__all__ = ['convert_mp3_to_wav']

MAX_PROB = -10000000000.0
FRAME_DURATION_IN_MS = 40


def get_segments(log_probs: np.ndarray,
                 path_wav: Union[PosixPath, str],
                 transcript_file: Union[PosixPath, str],
                 output_file:str,
                 vocabulary:List[str],
                 window_len:int)-> None:
    """
    Segments the audio into segments and saves segments timings to a file
    log_probs: Log probabilities for the original audio from an ASR model, shape T * |vocabulary|.
               values for blank should be at position 0
    path_wav: path to the audio .wav file
    transcript_file: path to
    output_file: path to the file to save timings for segments
    vocabulary: vocabulary used to train the ASR model, note blank is at position 0
    window_len: the length of each utterance (in terms of frames of the CTC outputs) fits into that window. The default window is 8000, your audio file is much shorter. You may reduce this value to improve alignment speed.
    """
    import pdb; pdb.set_trace()
    with open(transcript_file, "r") as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    logging.info(f"Syncing {transcript_file}")
    ground_truth_mat, utt_begin_indices = prepare_text(text, vocabulary)

    logging.info(
        f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
    )

    if len(ground_truth_mat) > log_probs.shape[0]:
        logging.warning(f"Skipping: Audio {path_wav} is shorter than text {transcript_file}")
    else:
        start_time = time.time()
        timings, char_probs, char_list = align(
            log_probs,
            vocabulary=vocabulary,
            ground_truth=ground_truth_mat,
            utt_begin_indices=utt_begin_indices,
            window_len=window_len,
        )

        total_time = time.time() - start_time
        logging.info(f"Time: {total_time}s ---> ~{round(total_time/60)}min")
        logging.info(f"Saving segments to {output_file}")

        write_output(output_file, utt_begin_indices, char_probs, path_wav, timings, text)


def align(lpz, vocabulary, ground_truth, utt_begin_indices, window_len, skip_prob=MAX_PROB):
    blank = 0

    # Try multiple window lengths if it fails
    while True:
        # Create table which will contain alignment probabilities
        table = np.zeros([min(window_len, lpz.shape[0]), len(ground_truth)], dtype=np.float32)
        table.fill(MAX_PROB)

        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int)

        # Run actual alignment
        start_time = time.time()
        t, c = cython_fill_table(
            table,
            lpz.astype(np.float32),
            np.array(ground_truth),
            offsets,
            np.array(utt_begin_indices),
            blank,
            skip_prob,
        )
        cython_fill_time = time.time() - start_time
        logging.info(f"Cython fill time: {cython_fill_time}s ---> ~{round(cython_fill_time/60)}min")
        logging.info("Max prob: " + str(table[:, c].max()) + " at " + str(t))

        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        state_list = [""] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probabilities towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = MAX_PROB
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = lpz[t + offsets[c], ground_truth[c, s]] if c > 0 else MAX_PROB
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s

                        max_lpz_prob = max(max_lpz_prob, switch_prob)

                stay_prob = max(lpz[t + offsets[c], blank], max_lpz_prob) if t > 0 else MAX_PROB
                est_stay_prob = table[t, c] - table[t - 1, c]

                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (offsets[c] + t) * 10 * 4 / 1000
                        char_probs[offsets[c] + t] = max_lpz_prob
                        state_list[offsets[c] + t] = vocabulary[ground_truth[c, min_s]]

                    c -= 1 + min_s
                    t -= 1 - offset

                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = "ε"
                    t -= 1
        except IndexError:
            # If the backtracking was not successful this usually means the window was too small
            window_len *= 2
            logging.info("IndexError: Trying with win len: " + str(window_len))
            if window_len < 100000:
                continue
            else:
                raise

        break

    return timings, char_probs, state_list


def prepare_text(text, char_list):
    """
    # Prepares the given text for alignment
    # Therefore we create a matrix of possible character symbols to represent the given text

    # Create list of char indices depending on the models char list
    """
    ground_truth = "#"
    utt_begin_indices = []
    for utt in text:
        utt = utt.strip()
        # Only one space in-between
        if ground_truth[-1] != " ":
            ground_truth += " "

        # Start new utterance remeber index
        utt_begin_indices.append(len(ground_truth) - 1)

        # Add chars of utterance
        for char in utt:
            if char.isspace():
                if ground_truth[-1] != " ":
                    ground_truth += " "
            elif char in char_list and char not in [
                ".",
                ",",
                "-",
                "?",
                "!",
                ":",
                "»",
                "«",
                ";",
                "'",
                "›",
                "‹",
                "(",
                ")",
            ]:
                ground_truth += char

    # Add space to the end
    if ground_truth[-1] != " ":
        ground_truth += " "
    utt_begin_indices.append(len(ground_truth) - 1)

    # Create matrix where first index is the time frame and the second index is the number of letters the character symbol spans
    max_char_len = max([len(c) for c in char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            if span in char_list:
                ground_truth_mat[i, s] = char_list.index(span)
    return ground_truth_mat, utt_begin_indices


def ind_to_text(ground_truth_mat, char_list):
    g = ground_truth_mat.flatten()
    text = ""
    for idx in g:
        if idx == -1:
            text += " "
        else:
            text += char_list[idx]
    logging.info("text len:", len(text))
    logging.info(text)


def write_output(out_path, utt_begin_indices, char_probs, path_wav, timings, text):
    dir_name = os.path.dirname(out_path)
    base_name = os.path.basename(path_wav)
    pickle.dump(utt_begin_indices, open(os.path.join(dir_name, base_name + "_utt_begin_indices.p"), "wb"))
    pickle.dump(char_probs, open(os.path.join(dir_name, base_name + "_char_probs.p"), "wb"))
    pickle.dump(timings, open(os.path.join(dir_name, base_name + "_timings.p"), "wb"))
    pickle.dump(text, open(os.path.join(dir_name, base_name + "_text.p"), "wb"))

    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        def compute_time(index, type, adj=2):
            # Compute start and end time of utterance.
            middle = (timings[index] + timings[index - 1]) / 2
            if type == "begin":
                return max(timings[index + 1] - 0.5, middle)
            elif type == "end":
                return min(timings[index - 1] + 0.5, middle)

        for i in range(len(text)):
            start = compute_time(utt_begin_indices[i], "begin")
            end = compute_time(utt_begin_indices[i + 1], "end")
            start_t = int(round(start * 1000 / FRAME_DURATION_IN_MS))
            end_t = int(round(end * 1000 / FRAME_DURATION_IN_MS))

            # Compute confidence score by using the min mean probability after splitting into segments of 30 frames
            n = 30
            if end_t == start_t:
                min_avg = 0
            elif start_t > end_t:
                min_avg = -100000
            elif end_t - start_t <= n:
                min_avg = char_probs[start_t:end_t].mean()
            else:
                min_avg = 0
                for t in range(start_t, end_t - n):
                    min_avg = min(min_avg, char_probs[t : t + n].mean())


            # from nemo.collections import asr as nemo_asr
            # import torch
            # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')
            # import pdb; pdb.set_trace()
            # preds = torch.tensor(char_probs[start_t:end_t]).unsqueeze(0)
            # tr = asr_model._wer.ctc_decoder_predictions_tensor(preds)
            # print (tr)
            # import pdb; pdb.set_trace()

            outfile.write(str(start) + " " + str(end) + " " + str(min_avg) + " | " + text[i] + "\n")


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
