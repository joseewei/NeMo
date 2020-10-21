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
import re
from pathlib import Path

format = '.mp3'
data_dir = '/home/ebakhturina/data/segmentation/librivox/ru/tolstoy/'
in_text = '/home/ebakhturina/data/segmentation/librivox/ru/tolstoy/childhood.txt'
output_dir = '/home/ebakhturina/data/segmentation/librivox/ru/tolstoy/processed'
os.makedirs(output_dir, exist_ok=True)

audio_paths = sorted(list(Path(data_dir).glob("*" + format)))

with open(in_text, 'r') as f:
    text = f.read()

split_pattern = "(\nГлава)"
text = re.split(split_pattern, text)

text_files = []
text_segment_id = 0
for i, audio in enumerate(sorted(audio_paths)):
    file_name = audio.name.replace(format, '.txt')
    if i == 0:
        # combine the first 3 segements - title + separator + the first chapter
        text_segment = text[:3]
        text_segment_id = 3
    else:
        text_segment = text[text_segment_id : text_segment_id + 2]
        text_segment_id = text_segment_id + 2
    text_files.append(file_name)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        f.write(''.join(text_segment))

# cut = re.split(r"(\nГлава)", text)
# print (len(cut))

if len(list(audio_paths)) != len(text_files):
    print(len(list(audio_paths)))
    print(len(text_files))
    import pdb

    pdb.set_trace()
    raise ValueError(f'Not correct split')
