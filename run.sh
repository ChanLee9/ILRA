DATASET_NAME=CoLA
TASK_TYPE=NLU
BATCH_SIZE=32
LR=2e-3
MAX_LEN=512
METHOD=pilra
LORA_R=2
KRONA_DIM=12
SCALING_ALPHA=1
NUM_EPOCHS=10
K=1
MODULES_TO_APPLY="query,value"
ABLATION=1

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
    --lora_r $LORA_R \
    --method $METHOD \
    --krona_dim $KRONA_DIM \
    --scaling_alpha $SCALING_ALPHA \
    --modules_to_apply $MODULES_TO_APPLY \
    --scaling_alpha $SCALING_ALPHA \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --do_test 0 \
    --scale 1 \
    --k $K \
    --ablation $ABLATION
