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
import multiprocessing
import os
import re
import string
from pathlib import Path

import scipy.io.wavfile as wavfile

from nemo.collections import asr as nemo_asr
from nemo.utils import logging

parser = argparse.ArgumentParser(description="Prepare transcript for segmentation")
parser.add_argument("--in_text", type=str, default=None, help='Path to input text file')
parser.add_argument("--output_dir", type=str, required=True, help='Path to output directory')
parser.add_argument("--audio_dir", type=str, help='Path to folder with .mp3 audio files')
parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling rate used for the model')
parser.add_argument('--language', type=str, default='eng', choices=['eng', 'ru'])
parser.add_argument(
    '--cut_prefix',
    type=int,
    default=0,
    help='Number of secs to from the beginning of the audio files. Librivox audio files contains long intro.',
)
parser.add_argument(
    '--model', type=str, default='QuartzNet15x5Base-En', help='Path to model checkpoint or ' 'pretrained model name'
)


LATIN_TO_RU = {
    'a': 'а',
    'b': 'б',
    'c': 'к',
    'd': 'д',
    'e': 'е',
    'f': 'ф',
    'g': 'г',
    'h': 'х',
    'i': 'и',
    'j': 'ж',
    'k': 'к',
    'l': 'л',
    'm': 'м',
    'n': 'н',
    'o': 'о',
    'p': 'п',
    'q': 'к',
    'r': 'р',
    's': 'с',
    't': 'т',
    'u': 'у',
    'v': 'в',
    'w': 'в',
    'x': 'к',
    'y': 'у',
    'z': 'з',
}
MISC_TO_RU = {
    'à': 'а',
    'è': 'е',
    'é': 'е',
    ' р.': ' рублей',
    ' к.': ' копеек',
    ' коп.': ' копеек',
    ' копек.': ' копеек',
    ' т.д. ': ' так далее ',
    ' т. д. ': ' так далее ',
    ' т.п. ': ' тому подобное ',
    ' т. п. ': ' тому подобное ',
    ' т.e. ': ' то есть ',
    ' т. e. ': ' то есть ',
}
NUMBERS_TO_ENG = {
    '0': 'zero ',
    '1': 'one ',
    '2': 'two ',
    '3': 'three ',
    '4': 'four ',
    '5': 'five ',
    '6': 'six ',
    '7': 'seven ',
    '8': 'eight ',
    '9': 'nine ',
}

NUMBERS_TO_RU = {
    '0': 'ноль ',
    '1': 'один ',
    '2': 'два ',
    '3': 'три ',
    '4': 'четыре ',
    '5': 'пять ',
    '6': 'шесть ',
    '7': 'семь ',
    '8': 'восемь ',
    '9': 'девять ',
}


def convert_mp3_to_wav(mp3_file: str, wav_file: str = None, sample_rate: int = 16000) -> str:
    """
    Converts .mp3 to .wav and changes sample rate if needed

    mp3_file: Path to .mp3 file
    sample_rate: Desired sample rate

    Returns:
        path to .wav file
    """
    logging.info(f"Converting {mp3_file} to .wav format with sample rate {sample_rate}")

    if wav_file is None:
        wav_file = mp3_file.replace(".mp3", ".wav")
    os.system(f'ffmpeg -i {mp3_file} -ac 1 -af aresample=resampler=soxr -ar {sample_rate} {wav_file} -y')
    return wav_file


def process_audio(mp3_file: str, wav_file: str = None, cut_prefix: int = 0, sample_rate: int = 16000):
    """Processes audio file: .mp3 to .wav conversion and cut a few seconds from the begging of the audio"""
    wav_audio = convert_mp3_to_wav(str(mp3_file), wav_file, sample_rate)

    # cut a few seconds of audio from the beginning
    sample_rate, signal = wavfile.read(wav_audio)
    wavfile.write(wav_audio, data=signal[cut_prefix * sample_rate :], rate=sample_rate)


def split_text(
    in_file: str, out_file: str, vocabulary=None, language='eng', remove_square_brackets=True, do_lower_case=True
):
    """
    Breaks down the in_file by sentences. Each sentence will be on a separate line.
    Also normalizes text: removes punctuation and applies lower case

    Args:
        in_file: path to original transcript
        out_file: file to the out file
    """

    logging.info(f'Splitting text in {in_file} into sentences.')
    with open(in_file, "r") as f:
        transcript = f.read()

    transcript = transcript.replace("\n", " ")

    if remove_square_brackets:
        transcript = re.sub(r'(\[.*?\])', ' ', transcript)
        logging.info(f'Removed text in [square] brackets')

    # Read and split transcript by utterance (roughly, sentences)
    split_pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s"
    sentences = re.split(split_pattern, transcript)

    # save split text with original punctuation and case
    out_dir, out_file_name = os.path.split(out_file)
    with open(os.path.join(out_dir, out_file_name[:-4] + '_with_punct.txt'), "w") as f:
        sentences_with_punct = "\n".join([s for s in sentences if s])
        # remove extra space
        sentences_with_punct = re.sub(r' +', ' ', sentences_with_punct)
        f.write(sentences_with_punct)

    # make sure to leave punctuation present in vocabulary
    all_punct_marks = string.punctuation
    if vocabulary:
        for v in vocabulary:
            all_punct_marks = all_punct_marks.replace(v, '')
    sentences = [re.sub("[" + all_punct_marks + "]", "", t).strip() for t in sentences]
    sentences = "\n".join([s for s in sentences if s])

    if do_lower_case:
        sentences = sentences.lower()

    if language == 'eng':
        # remove text in square brackets - translation
        for k, v in NUMBERS_TO_ENG.items():
            sentences = sentences.replace(k, v)
        # remove non acsii characters
        sentences = ''.join(i for i in sentences if ord(i) < 128)
    elif language == 'ru':
        if vocabulary and '-' not in vocabulary:
            sentences.replace('-', ' ')
        # remove text in square brackets - translation
        for k, v in NUMBERS_TO_RU.items():
            sentences = sentences.replace(k, v)
        # repalce latin charaters with russian
        for k, v in LATIN_TO_RU.items():
            sentences = sentences.replace(k, v)

    # remove extra space
    sentences = re.sub(r' +', ' ', sentences)
    with open(out_file, "w") as f:
        f.write(sentences)


if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    text_files = []
    if args.in_text:
        vocabulary = None
        if args.model is None:
            logging.info(f'No model provided, model vocabulary wont be used')
        elif os.path.exists(args.model):
            asr_model = nemo_asr.models.EncDecCTCModel.restore_from(args.model)
            vocabulary = asr_model.cfg.decoder['params']['vocabulary']
        elif args.model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
            asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(args.model)
            vocabulary = asr_model.cfg.decoder['params']['vocabulary']
        else:
            logging.info(
                f'Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.list_available_models()}'
            )

        if os.path.isdir(args.in_text):
            text_files = Path(args.in_text).glob(("*.txt"))
        else:
            text_files.append(Path(args.in_text))
        for text in text_files:
            base_name = os.path.basename(text)[:-4]
            out_text_file = os.path.join(args.output_dir, base_name + '.txt')

            split_text(text, out_text_file, vocabulary=vocabulary, language=args.language)
            logging.info(f'Text saved to {out_text_file}')

        if args.audio_dir:
            if not os.path.exists(args.audio_dir):
                raise ValueError(f'Provide a valid path to the audio files, provided: {args.audio_dir}')
            audio_paths = list(Path(args.audio_dir).glob("*.mp3"))

            workers = []
            for i in range(len(audio_paths)):
                wav_file = os.path.join(args.output_dir, audio_paths[i].name.replace(".mp3", ".wav"))
                worker = multiprocessing.Process(
                    target=process_audio, args=(audio_paths[i], wav_file, args.cut_prefix, args.sample_rate),
                )
                workers.append(worker)
                worker.start()
            for w in workers:
                w.join()

    logging.info('Done.')
