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

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import scipy.io.wavfile as wav
import torch
from prepare_data import convert_mp3_to_wav
from utils import get_segments, listener_configurer, listener_process, worker_configurer, worker_process

import nemo.collections.asr as nemo_asr

parser = argparse.ArgumentParser(description="CTC Segmentation")
parser.add_argument("--output_dir", default='output', type=str, help='Path to output directory')
parser.add_argument(
    "--data",
    default='/home/ebakhturina/data/segmentation/sample/nv_test.wav',
    type=str,
    help='Path to directory with audio files and associated transcripts (same respective names only formats are '
    'different or path to wav file (transcript should have the same base name and be located in the same folder'
    'as the wav file.',
)
parser.add_argument('--window_len', type=int, default=8000)
parser.add_argument('--format', type=str, default='.wav', choices=['.wav', '.mp3'])
parser.add_argument('--no_parallel', action='store_true')
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument(
    '--model', type=str, default='QuartzNet15x5Base-En', help='Path to model checkpoint or ' 'pretrained model name'
)
parser.add_argument('--debug', action='store_true', help='Set to True for debugging')

if __name__ == '__main__':
    args = parser.parse_args()

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'ctc_segmentation.log')
    level = 'DEBUG' if args.debug else 'INFO'
    if args.no_parallel:
        logger = logging.getLogger('CTC')
        file_handler = logging.FileHandler(filename=log_file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(handlers=handlers, level=level)
    else:
        queue = multiprocessing.Queue(-1)

        listener = multiprocessing.Process(target=listener_process, args=(queue, listener_configurer, log_file, level))
        worker_configurer(queue, level)
        listener.start()

    if os.path.exists(args.model):
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(args.model)
    elif args.model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(args.model, strict=False)
    else:
        raise ValueError(
            f'Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.list_available_models()}'
        )

    # extract ASR vocabulary and add blank symbol
    vocabulary = asr_model.cfg.decoder['params']['vocabulary']
    odim = len(asr_model._cfg.decoder.params['vocabulary']) + 1
    logging.debug(f'ASR Model vocabulary: {vocabulary}')

    # add blank to vocab
    vocabulary = ["ε"] + list(vocabulary)
    data = Path(args.data)
    output_dir = Path(args.output_dir)

    if os.path.isdir(data):
        audio_paths = data.glob("*" + args.format)
        data_dir = data
    else:
        audio_paths = [Path(data)]
        data_dir = Path(os.path.dirname(data))

    all_log_probs = []
    all_transcript_file = []
    all_segment_file = []
    all_wav_paths = []

    for path_audio in audio_paths:
        if args.format == ".mp3":
            path_audio = Path(convert_mp3_to_wav(str(path_audio), args.sampling_rate))

        transcript_file = data_dir / path_audio.name.replace(".wav", ".txt")
        segment_file = output_dir / path_audio.name.replace(".wav", "_segments.txt")

        try:
            sampling_rate, signal = wav.read(path_audio)
            if sampling_rate != args.sampling_rate:
                logging.info(f'Converting {path_audio} from {sampling_rate} to {args.sampling_rate}')
                start_time = time.time()
                signal, sampling_rate = librosa.load(path_audio, sr=args.sampling_rate)
                logging.info(f'Time to convert {time.time() - start_time}')

        except ValueError:
            logging.error(f"Check that '--format .mp3' arg is used for .mp3 audio files")
            raise

        original_duration = len(signal) / sampling_rate
        logging.debug(f'Duration: {original_duration}s, file_name: {path_audio}')

        log_probs = asr_model.transcribe(paths2audio_files=[str(path_audio)], batch_size=1, logprobs=True)[0].cpu()

        # move blank values to the first column
        log_probs = np.squeeze(log_probs, axis=0)
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
        all_log_probs.append(log_probs)
        all_segment_file.append(str(segment_file))
        all_transcript_file.append(str(transcript_file))
        all_wav_paths.append(path_audio)

    del asr_model
    torch.cuda.empty_cache()

    start_time = time.time()
    if args.no_parallel:
        for i in range(len(all_log_probs)):
            get_segments(
                all_log_probs[i],
                all_wav_paths[i],
                all_transcript_file[i],
                all_segment_file[i],
                vocabulary,
                args.window_len,
            )
    else:
        workers = []
        for i in range(len(all_log_probs)):
            worker = multiprocessing.Process(
                target=worker_process,
                args=(
                    queue,
                    worker_configurer,
                    level,
                    all_log_probs[i],
                    all_wav_paths[i],
                    all_transcript_file[i],
                    all_segment_file[i],
                    vocabulary,
                    args.window_len,
                ),
            )
            workers.append(worker)
            worker.start()
        for w in workers:
            w.join()
        queue.put_nowait(None)
        listener.join()

    total_time = time.time() - start_time
    logging.info(f'Total execution time: ~{round(total_time/60)}min')
    logging.info(f'Saving logs to {log_file}')
