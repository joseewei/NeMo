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
    "--audio_file", type=str, required=True, help='Path to original .wav file used for the ctc-segmentation',
)
parser.add_argument(
    "--alignment",
    type=str,
    required=True,
    help='Path to a .txt file with timestamps - result of the ctc-segmentation',
)
parser.add_argument("--threshold", type=float, default=-5, help='Minimum score value accepted')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

fragments_dir = os.path.join(args.output_dir, "wav")
os.makedirs(fragments_dir, exist_ok=True)


sr, x = wavfile.read(args.audio_file)
base_dir, base_name = os.path.split(args.alignment)
base_name = base_name[:-4]

segments = []
ref_text = []
logging.info(f'Using {args.alignment}')
with open(args.alignment, 'r') as f:
    for line in f:
        line = line.split('|')
        # skip first line
        if len(line) == 1:
            continue
        text = line[1]
        line = line[0].split()
        segments.append((float(line[0]), float(line[1]), float(line[2])))
        ref_text.append(text.strip())

original_duration = len(x) / sr
logging.info(f'Original duration: {round(original_duration)}s or ~{round(original_duration/60)}min')
manifest_path = os.path.join(args.output_dir, f'{base_name}_manifest.json')

low_score_segments_duration = 0

low_score_segments_dir = os.path.join(args.output_dir, "low_score")
os.makedirs(low_score_segments_dir, exist_ok=True)
low_score_segments_manifest = os.path.join(args.output_dir, f'{base_name}_low_score_manifest.json')
total_dur = 0
start = 0
with open(manifest_path, 'w') as f:
    with open(low_score_segments_manifest, 'w') as low_score_f:
        for i, (st, end, score) in enumerate(segments):
            segment = x[int(st * sr) : int(end * sr)]
            duration = len(segment) / sr
            if duration > 0:
                text = str(round(score, 2)) + '~ ' + ref_text[i]
                if score > args.threshold:
                    total_dur += duration
                    audio_filepath = os.path.join(fragments_dir, f'{base_name}_{i:03}.wav')
                    file_to_write = f

                    wavfile.write(audio_filepath, sr, segment)
                    info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': text}
                    json.dump(info, f)
                    f.write('\n')
                else:
                    low_score_segments_duration += duration
                    audio_filepath = os.path.join(low_score_segments_dir, f'{base_name}_{i:03}.wav')
                    file_to_write = low_score_f

                wavfile.write(audio_filepath, sr, segment)
                info = {
                    'audio_filepath': audio_filepath,
                    'duration': duration,
                    'text': text,
                }
                json.dump(info, low_score_f)
                low_score_f.write('\n')

logging.info(f'Saved files duration: {round(total_dur)}s or ~{round(total_dur/60)}min at {args.output_dir}')
logging.info(
    f'Low score segments duration: {round(low_score_segments_duration)}s or ~{round(low_score_segments_duration/60)}min saved at {low_score_segments_dir}'
)

# save deleted segments along with manifest
deleted = []
del_fragments = os.path.join(args.output_dir, 'deleted')
os.makedirs(del_fragments, exist_ok=True)
del_manifest = os.path.join(args.output_dir, f'{base_name}_del_manifest.json')
del_duration = 0
begin = 0
with open(del_manifest, 'w') as f:
    for i, (st, end, _) in enumerate(segments):
        if st - begin > 0.01:
            segment = x[int(begin * sr) : int(st * sr)]
            audio_filepath = os.path.join(del_fragments, f'del_{base_name}_{i:03}.wav')
            wavfile.write(audio_filepath, sr, segment)
            duration = len(segment) / sr
            del_duration += duration
            deleted.append((begin, st))
            info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': 'n/a'}
            json.dump(info, f)
            f.write('\n')
        begin = end

    segment = x[int(begin * sr) :]
    audio_filepath = os.path.join(del_fragments, f'del_{i+1:03}.wav')
    wavfile.write(audio_filepath, sr, segment)
    duration = len(segment) / sr
    del_duration += duration
    deleted.append((begin, original_duration))

    info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': 'n/a'}
    json.dump(info, f)
    f.write('\n')

logging.info(f'Saved DEL files duration: {round(del_duration)}s or ~ {round(del_duration/60)}min at {del_fragments}')
missing_audio = original_duration - total_dur - del_duration - low_score_segments_duration
logging.info(f'Missing audio: {round(missing_audio)}s or ~{round((missing_audio)/60)}min')
