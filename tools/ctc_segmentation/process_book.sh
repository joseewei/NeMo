#!/bin/bash

BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv'
OUTPUT_DIR=$BASE_DIR/v4
MODEL='/home/ebakhturina/nemo_ckpts/quartznet/QuartzNet15x5-Ru-e512-wer14.45.nemo'

python cut_text_into_chapters_goncharov.py

python prepare_data.py \
--in_text=$BASE_DIR/chapters/ \
--output_dir=$OUTPUT_DIR/processed/ \
--language='ru' \
--cut_prefix=3 \
--model=$MODEL \
--audio_dir=$BASE_DIR/audio/ || exit

for WINDOW in 8000 10000 12000
do
  python run_ctc_segmentation.py \
  --output_dir=$OUTPUT_DIR \
  --data=$OUTPUT_DIR/processed/ \
  --model=$MODEL  --window_len $WINDOW || exit
done

 verify aligned segments
python verify_segments.py \
--base_dir=$OUTPUT_DIR  || exit

python cut_audio.py \
--output_dir=$OUTPUT_DIR \
--model=$MODEL \
--alignment=$OUTPUT_DIR/verified_segments || exit

python process_manifests.py \
--output_dir=$OUTPUT_DIR \
--manifests_dir=$OUTPUT_DIR/manifests/ \
--num_samples 6



