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
from pathlib import Path

from scipy.io import wavfile

from nemo.utils import logging

parser = argparse.ArgumentParser(description="Cut audio on the segments based on segments")
parser.add_argument("--output_dir", default='output', type=str, help='Path to output directory')
parser.add_argument(
    "--alignment", type=str, required=True, help='Path to book.aligned file',
)
args = parser.parse_args()

base_dir = Path(os.path.dirname(args.alignment))
wav_paths = sorted(base_dir.glob('*.wav'))
orig_signals = []
orig_srs = []
orig_durations = []

for wav_path in wav_paths:
    sr, signal = wavfile.read(wav_path)
    orig_signals.append(signal)
    orig_durations.append(len(signal) / sr)
    orig_srs.append(sr)

with open(args.alignment, 'r') as f:
    lines = f.read()
separator = '@@'
lines = lines[1:-1].replace('{"start":', separator + '{"start":').split(separator)
lines = [l for l in lines if l]

os.makedirs(args.output_dir, exist_ok=True)
wav_dir = os.path.join(args.output_dir, "wav")
os.makedirs(wav_dir, exist_ok=True)
manifest_path = os.path.join(args.output_dir, 'manifest.json')

total_durations = [0] * len(orig_durations)

with open(manifest_path, 'w') as f:
    for i, line in enumerate(lines):
        line = line.strip()
        if line[-1] == ',':
            line = line[:-1]
        line = json.loads(line)
        section_id = line['meta']['section'][0]
        signal = orig_signals[section_id]
        sr = orig_srs[section_id]
        start = int(line['start'] * sr / 1000)
        end = int(line['end'] * sr / 1000)

        segment = signal[start:end]
        duration = len(segment) / sr
        if duration > 0:
            text = (
                'cer: ' + str(round(line['cer'], 1)) + ' Raw: ' + line['aligned-raw'] + ' ~Tr: ' + line['transcript']
            )
            total_durations[section_id] += duration
            audio_filepath = os.path.join(wav_dir, f'{section_id}_{i:04}.wav')
            wavfile.write(audio_filepath, sr, segment)
            info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': text}
            json.dump(info, f)
            f.write('\n')

for i in range(len(total_durations)):
    logging.info(
        f'{i} Original duration   : {round(orig_durations[i])}s or ~{round(orig_durations[i] / 60)}min at {wav_dir}'
    )
    logging.info(
        f'{i} Saved files duration: {round(total_durations[i])}s or ~{round(total_durations[i]/60)}min at {wav_dir}'
    )
