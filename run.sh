DARASET_NAME=MRPC
TASK_TYPE=NLU
BATCH_SIZE=4
LR=2e-5
MAX_LEN=512
METHOD=bit_fit
LORA_R=2

python main.py \
    --dataset_name $DARASET_NAME \
    --model_name_or_path ../pretrained_models/roberta \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LEN \
    --lr $LR \
    --num_epochs 3 \
    --task_type $TASK_TYPE \
    --device cuda:0 \
    --num_tags 2 \
    --modules_to_apply $METHOD \
    --lora_r $LORA_R \
    --lora_alpha 8 \
    --method $METHOD \
    --dropout 0.1 \


