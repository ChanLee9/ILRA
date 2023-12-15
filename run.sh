DATASET_NAME=MRPC
TASK_TYPE=NLU
BATCH_SIZE=64
LR=1e-3
MAX_LEN=512
METHOD=ilra
LORA_R=2
KRONA_DIM=32
SCALING_ALPHA=2
NUM_EPOCHS=10
MODULES_TO_APPLY="query,key,value,output,ffn1,ffn2"

python main.py \
    --dataset_name $DATASET_NAME \
    --model_name_or_path ../pretrained_models/roberta \
    --task_type $TASK_TYPE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_length $MAX_LEN \
    --num_epochs $NUM_EPOCHS \
    --method $METHOD \
    --device cuda:0 \
    --num_tags 2 \
    --lora_r $LORA_R \
    --method $METHOD \
    --krona_dim $KRONA_DIM \
    --scaling_alpha $SCALING_ALPHA \
    --modules_to_apply $MODULES_TO_APPLY \
    --scaling_alpha $SCALING_ALPHA \
    --dropout 0.1 \
    --weight_decay 1e-2 \
    --warmup_steps 100 \
