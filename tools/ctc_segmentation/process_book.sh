python run_ctc_segmentation.py \
--output_dir /home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/debug/ \
--data /home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/debug/obryv_001_goncharov_64kb.wav \
--model /home/ebakhturina/nemo_ckpts/quartznet/QuartzNet15x5-Ru-e512-wer14.45.nemo --window_len 4000

python cut_audio.py \
--output_dir /home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/debug/ \
--model /home/ebakhturina/nemo_ckpts/quartznet/QuartzNet15x5-Ru-e512-wer14.45.nemo \
--alignment /home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/debug/obryv_001_goncharov_64kb_segments.txt
