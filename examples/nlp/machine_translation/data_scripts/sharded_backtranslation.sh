PYTHONUNBUFFERED=1
NUM_SHARDS=8
NUM_GPUS=7
mono_data=$1
output_path=$2
model_path=$3

echo "Sharding ${mono_data} ..."

wc -l ${mono_data}

split -n ${NUM_SHARDS} -d -a 2 ${mono_data} /tmp/shard

echo "Launching backtranslation with ${model_path}"

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

wait