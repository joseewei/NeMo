#!/bin/bash

MODEL='/home/ebakhturina/nemo_ckpts/quartznet/QuartzNet15x5-Ru-e512-wer14.45.nemo'

# BOOK 01
BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv'

OFFSET=-50
OUTPUT_DIR=$BASE_DIR/v5_$OFFSET


#python /home/ebakhturina/scripts/ctc_segmentation/divide_into_chapters/cut_text_into_chapters_goncharov.py  \
#--audio_format = '.mp3' \
#--audio_data_dir = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/audio' \
#--in_text = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/01_goncharov.txt' \
#--output_dir = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/chapters'


#python prepare_data.py \
#--in_text=$BASE_DIR/chapters/ \
#--output_dir=$OUTPUT_DIR/processed/ \
#--language='ru' \
#--cut_prefix=3 \
#--model=$MODEL \
#--audio_dir=$BASE_DIR/audio/ || exit

## BOOK 02
#BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/02_AYKHENVALD/'
#OUTPUT_DIR=$BASE_DIR/v1
#python prepare_data.py \
#--in_text=$BASE_DIR/data/text/ \
#--output_dir=$OUTPUT_DIR/processed/ \
#--language='ru' \
#--cut_prefix=3 \
#--model=$MODEL \
#--audio_dir=$BASE_DIR/data/audio/ || exit
#
## BOOK 04
#BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/04_obyknovennaya-istorya-by-ivan-goncharov'
#OUTPUT_DIR=$BASE_DIR/v1
#OFFSET=0
#python cut_text_into_chapters_goncharov.py  \
#--audio_format='.mp3' \
#--audio_data_dir=$BASE_DIR/data/audio \
#--in_text=$BASE_DIR/data/04_obyknovennaya-istorya-by-ivan-goncharov.txt \
#--output_dir=$BASE_DIR/chapters
#
## BOOK 05
#BASE_DIR='/home/ebakhturina/data/segmentation/librivox/ru/07_garshin'
#OUTPUT_DIR=$BASE_DIR/v1
#OFFSET=0
#
#python prepare_data.py \
#--in_text=$BASE_DIR/chapters/ \
#--output_dir=$OUTPUT_DIR/processed/ \
#--language='ru' \
#--cut_prefix=3 \
#--model=$MODEL \
#--audio_dir=$BASE_DIR/data/audio/ || exit


#for WINDOW in 8000 10000 12000
#do
#  python run_ctc_segmentation.py \
#  --output_dir=$OUTPUT_DIR \
#  --data=$OUTPUT_DIR/processed/ \
#  --model=$MODEL  --window_len $WINDOW || exit
#done
#
## verify aligned segments
#python verify_segments.py \
#--base_dir=$OUTPUT_DIR  || exit
#
#python cut_audio.py \
#--output_dir=$OUTPUT_DIR \
#--model=$MODEL \
#--alignment=$OUTPUT_DIR/verified_segments \
#--offset=$OFFSET || exit
#
#python process_manifests.py \
#--output_dir=$OUTPUT_DIR \
#--manifests_dir=$OUTPUT_DIR/manifests/ \
#--num_samples 6

exit

