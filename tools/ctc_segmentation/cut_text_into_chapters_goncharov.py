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
audio_data_dir = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/audio'
in_text = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/sublime_goncharov.txt'
output_dir = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/chapters'
os.makedirs(output_dir, exist_ok=True)


audio_paths = sorted(list(Path(audio_data_dir).glob("*" + format)))

with open(in_text, 'r') as f:
    text = f.read()

pattern = "\n\n[A-Z]+\n\n"
chapter_romans = re.findall(pattern, text)

assert len(chapter_romans) == len(audio_paths), 'Incorrect split'

ru_zero = ['первая', 'вторая', 'третья', 'четвертая', 'пятая', 'шестая', 'седьмая', 'восьмая', 'девятая', 'десятая']
ru_first = [
    'одиннадцатая',
    'двенадцатая',
    'тринадцатая',
    'четырнадцатая',
    'пятнадцатая',
    'шестнадцатая',
    'семнадцатая',
    'восемнадцатая',
    'девятнадцатая',
]
ru_second = ['двадцатая'] + ['двадцать ' + n for n in ru_zero[:-1]]
ru_third = ['тридцатая'] + ['тридцать ' + n for n in ru_zero[:-1]]
ru_fourth = ['сороковая'] + ['сорок ' + n for n in ru_zero[:-1]]
ru_text = ru_zero + ru_first + ru_second + ru_third + ru_fourth

roman_zero = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
roman_first = ['X'] + ['X' + r for r in roman_zero[:-1]]
roman_second = ['XX'] + ['XX' + r for r in roman_zero[:-1]]
roman_third = ['XXX'] + ['XX' + r for r in roman_zero[:-1]]
roman_fourth = ['XL'] + ['XL' + r for r in roman_zero[:-1]]
roman_numerals = roman_zero + roman_first + roman_second + roman_third + roman_fourth

text = re.split("(" + pattern + ")", text)
replaced = 0
del_pattern = '##DELETE##'
for i, t in enumerate(text):
    if t.strip() in roman_numerals:
        idx = roman_numerals.index(t.strip())
        # removing chapter name since some audio parts have chapter name name followed by intro
        # that could break alignment
        text[i] = t.replace(roman_numerals[idx], 'Глава ' + ru_text[idx] + del_pattern)
        replaced += 1

text_files = []
text_segment_id = 0
for i, audio in enumerate(sorted(audio_paths)):
    file_name = audio.name.replace(format, '.txt')
    if i == 0:
        # combine the first 3 segments - title + separator + the first chapter
        text_segment = text[:3]
        text_segment_id = 3
    else:
        text_segment = text[text_segment_id : text_segment_id + 2]
        text_segment_id = text_segment_id + 2
    text_files.append(file_name)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        text_segment = ''.join(text_segment)
        # remove the name of the chapter
        del_idx = text_segment.index(del_pattern)
        text_segment = text_segment[del_idx + len(del_pattern) :]
        f.write(text_segment)

if len(list(audio_paths)) != len(text_files):
    raise ValueError(f'Incorrect split')
