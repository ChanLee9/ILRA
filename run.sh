DARASET_NAME=MRPC
TASK_TYPE=NLU
BATCH_SIZE=4
LR=2e-5
MAX_LEN=512

python main.py \
    --dataset_name $DARASET_NAME \
    --model_name_or_path ../pretrained_models/roberta \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LEN \
    --lr $LR \
    --num_epochs 3 \
    --task_type $TASK_TYPE \
    --device cuda:0 \
    --num_tags 2
