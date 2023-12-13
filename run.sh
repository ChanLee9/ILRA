DARASET_NAME=MRPC
TASK_TYPE=NLU
BATCH_SIZE=32
LR=1e-3
MAX_LEN=512
METHOD=krona
LORA_R=2
KRONA_DIM=4
SCALING_ALPHA=5

python main.py \
    --dataset_name $DARASET_NAME \
    --model_name_or_path ../pretrained_models/roberta \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LEN \
    --lr $LR \
    --num_epochs 5 \
    --task_type $TASK_TYPE \
    --device cuda:0 \
    --num_tags 2 \
    --lora_r $LORA_R \
    --method $METHOD \
    --krona_dim $KRONA_DIM \
    --scaling_alpha $SCALING_ALPHA \
    --dropout 0.1 \
    --weight_decay 1e-2 \


