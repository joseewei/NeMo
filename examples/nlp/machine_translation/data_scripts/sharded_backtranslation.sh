PYTHONUNBUFFERED=1
NUM_SHARDS=16
NUM_GPUS=15
mono_data=$1
output_path=$2
model_path=$3

split -n ${NUM_SHARDS} -d -a 2 ${mono_data} /tmp/shard

for i in $(seq 0 $NUM_GPUS); do
    if [ $i -gt 9 ]
        then
        suffix=$i
    else
        suffix=0$i
    fi
    CUDA_VISIBLE_DEVICES=${i} python ../generate_noisy_backtranslation_data.py \
        --model ${model_path} \
        --text2translate /tmp/shard${suffix} \
        --output ${output_path}/backtranslation.shard.${i} &
done
