#!/bin/bash

BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv'
OUTPUT_DIR=$BASE_DIR/v2_0_offset
MODEL='/home/ebakhturina/nemo_ckpts/quartznet/QuartzNet15x5-Ru-e512-wer14.45.nemo'

python cut_text_into_chapters_general.py

python prepare_data.py \
--in_text=$BASE_DIR/chapters/ \
--output_dir=$OUTPUT_DIR/processed/ \
--language='ru' \
--cut_prefix=3 \
--audio_dir=$BASE_DIR/audio/ \
--model=$MODEL || exit

python run_ctc_segmentation.py \
--output_dir=$OUTPUT_DIR/segments \
--data=$OUTPUT_DIR/processed/ \
--model=$MODEL  --window_len 8000 || exit

python cut_audio.py \
--output_dir=$OUTPUT_DIR \
--model=$MODEL \
--alignment=$OUTPUT_DIR/segments || exit


