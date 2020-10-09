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
import json
import os

from scipy.io import wavfile

from nemo.utils import logging

parser = argparse.ArgumentParser(description="Cut audio on the segments based on segments")
parser.add_argument("--output_dir", default='output', type=str, help='Path to output directory')
parser.add_argument(
    "--alignment",
    type=str,
    required=True,
    help='Path to a .txt file with timestamps - result of the ctc-segmentation',
)
parser.add_argument("--threshold", type=float, default=-5, help='Minimum score value accepted')

args = parser.parse_args()

# create directories to store high score .wav audio fragments, low scored once, and deleted
base_name = os.path.basename(args.alignment)[:-4]

os.makedirs(args.output_dir, exist_ok=True)
fragments_dir = os.path.join(args.output_dir, "segment_high_score")
del_fragments = os.path.join(args.output_dir, 'deleted')
low_score_segments_dir = os.path.join(args.output_dir, "low_score")

os.makedirs(fragments_dir, exist_ok=True)
os.makedirs(low_score_segments_dir, exist_ok=True)
os.makedirs(del_fragments, exist_ok=True)

manifest_path = os.path.join(args.output_dir, f'{base_name}_manifest.json')
low_score_segments_manifest = os.path.join(args.output_dir, f'{base_name}_low_score_manifest.json')
del_manifest = os.path.join(args.output_dir, f'{base_name}_del_manifest.json')

segments = []
ref_text = []
with open(args.alignment, 'r') as f:
    for line in f:
        line = line.split('|')
        # read audio file name from the first line
        if len(line) == 1:
            audio_file = line[0].strip()
            continue
        text = line[1]
        line = line[0].split()
        segments.append((float(line[0]), float(line[1]), float(line[2])))
        ref_text.append(text.strip())

sampling_rate, signal = wavfile.read(audio_file)
original_duration = len(signal) / sampling_rate
logging.info(f'Cutting {audio_file} based on {args.alignment}')
logging.info(f'Original duration: {round(original_duration)}s or ~{round(original_duration/60)}min')

low_score_segments_duration = 0
total_dur = 0
start = 0
with open(manifest_path, 'w') as f:
    with open(low_score_segments_manifest, 'w') as low_score_f:
        for i, (st, end, score) in enumerate(segments):
            segment = signal[int(st * sampling_rate) : int(end * sampling_rate)]
            duration = len(segment) / sampling_rate
            if duration > 0:
                text = str(round(score, 2)) + '~ ' + ref_text[i]
                if score > args.threshold:
                    total_dur += duration
                    audio_filepath = os.path.join(fragments_dir, f'{base_name}_{i:04}.wav')
                    file_to_write = f

                    wavfile.write(audio_filepath, sampling_rate, segment)
                    info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': text}
                    json.dump(info, f)
                    f.write('\n')
                else:
                    low_score_segments_duration += duration
                    audio_filepath = os.path.join(low_score_segments_dir, f'{base_name}_{i:04}.wav')
                    file_to_write = low_score_f

                wavfile.write(audio_filepath, sampling_rate, segment)
                info = {
                    'audio_filepath': audio_filepath,
                    'duration': duration,
                    'text': text,
                }
                json.dump(info, file_to_write)
                file_to_write.write('\n')

logging.info(f'Saved files duration: {round(total_dur)}s or ~{round(total_dur/60)}min at {args.output_dir}')
logging.info(
    f'Low score segments duration: {round(low_score_segments_duration)}s or ~{round(low_score_segments_duration/60)}min saved at {low_score_segments_dir}'
)

# save deleted segments along with manifest
deleted = []
del_duration = 0
begin = 0
with open(del_manifest, 'w') as f:
    for i, (st, end, _) in enumerate(segments):
        if st - begin > 0.01:
            segment = signal[int(begin * sampling_rate) : int(st * sampling_rate)]
            audio_filepath = os.path.join(del_fragments, f'del_{base_name}_{i:03}.wav')
            wavfile.write(audio_filepath, sampling_rate, segment)
            duration = len(segment) / sampling_rate
            del_duration += duration
            deleted.append((begin, st))
            info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': 'n/a'}
            json.dump(info, f)
            f.write('\n')
        begin = end

    segment = signal[int(begin * sampling_rate) :]
    audio_filepath = os.path.join(del_fragments, f'del_{i+1:03}.wav')
    wavfile.write(audio_filepath, sampling_rate, segment)
    duration = len(segment) / sampling_rate
    del_duration += duration
    deleted.append((begin, original_duration))

    info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': 'n/a'}
    json.dump(info, f)
    f.write('\n')

logging.info(f'Saved DEL files duration: {round(del_duration)}s or ~ {round(del_duration/60)}min at {del_fragments}')
missing_audio = original_duration - total_dur - del_duration - low_score_segments_duration
if missing_audio > 15:
    raise ValueError(f'{round(missing_audio)}s or ~ {round(missing_audio/60)}min is missing. Check the args')
