#!/bin/bash

MODEL_NAME_OR_PATH=$1
DATA_DIR=$2
OUTPUT_DIR=$3
OFFSET=0
CUT_PREFIX=3

SCRIPTS_DIR='scripts'

if [[ -z $1 ]] || [[ -z $2 ]] || [[ -z $3 ]]; then
  echo "Usage: $(basename "$0") [model_name_or_path] [data_dir] [output_dir]"
  exit 1
fi

# STEP #1
# Prepare text and audio data for segmentation
python $SCRIPTS_DIR/prepare_data.py \
--in_text=$DATA_DIR \
--output_dir=$OUTPUT_DIR/processed/ \
--language='eng' \
--cut_prefix=$CUT_PREFIX \
--model=$MODEL_NAME_OR_PATH \
--audio_dir=$DATA_DIR || exit

# STEP #2
# Run CTC-segmenatation
# example on how to perform alignemnt with various window sizes
# note if the alignment with the initial window size won't be found, the window size will be double to re-attempt
for WINDOW in 8000 10000 12000
do
  python $SCRIPTS_DIR/run_ctc_segmentation.py \
  --output_dir=$OUTPUT_DIR \
  --data=$OUTPUT_DIR/processed/ \
  --model=$MODEL_NAME_OR_PATH  \
  --window_len $WINDOW || exit
done

# STEP #3 (Optional)
# Verify aligned segments only if multiple WINDOWs used in the Step #2)
python $SCRIPTS_DIR/verify_segments.py \
--base_dir=$OUTPUT_DIR  || exit

# STEP #4
# Cut the original audio files based on the alignments
# (use --alignment=$OUTPUT_DIR/segments if only 1 WINDOW size was used in the Step #2)
# 3 manifests and coresponding clips folders will be created:
# - high scored clips
# - low scored clips
# - deleted segments
python $SCRIPTS_DIR/cut_audio.py \
--output_dir=$OUTPUT_DIR \
--model=$MODEL_NAME_OR_PATH \
--alignment=$OUTPUT_DIR/verified_segments \
--offset=$OFFSET || exit

# STEP #5 (Optional)
# If multiple audio files were segmented in the step #2, this step will aggregate manifests for high scored segments
# for all audio files into all_manifest.json
# Also a separate manifest with samples from across all high scored segments will credated if --num_samples > 0
# --num_samples samples will be taken from the beginning, end and the middle of the each audio file manifest and
# will be stored at sample_manifest.json
python $SCRIPTS_DIR/process_manifests.py \
--output_dir=$OUTPUT_DIR \
--manifests_dir=$OUTPUT_DIR/manifests/ \
--num_samples 0

exit

