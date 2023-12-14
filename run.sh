DARASET_NAME=MRPC
TASK_TYPE=NLU
BATCH_SIZE=64
LR=1e-3
MAX_LEN=512
METHOD=krona
LORA_R=2
KRONA_DIM=32
SCALING_ALPHA=16
NUM_EPOCHS=20
MODULES_TO_APPLY="query,key,value,output,ffn1,ffn2"

python main.py \
    --dataset_name $DARASET_NAME \
    --model_name_or_path ../pretrained_models/roberta \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LEN \
    --lr $LR \
    --num_epochs $NUM_EPOCHS \
    --task_type $TASK_TYPE \
    --device cuda:0 \
    --num_tags 2 \
    --lora_r $LORA_R \
    --method $METHOD \
    --krona_dim $KRONA_DIM \
    --scaling_alpha $SCALING_ALPHA \
    --dropout 0.1 \
    --weight_decay 1e-2 \
    --warmup_steps 100 \
    --modules_to_apply $MODULES_TO_APPLY \


