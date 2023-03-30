#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate p37

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=$1
MASTER_PORT=6000
NNODES=$2
NODE_RANK=$3
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=~/data/my-bert_text_sentence

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 4 \
       --hidden-size 128 \
       --num-attention-heads 16 \
       --micro-batch-size 32 \
       --global-batch-size 1024 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 16000 \
       --data-path $DATA_PATH \
       --vocab-file ~/data/bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1.4e-2 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 15840 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .02 \
       --log-interval 40 \
       --eval-interval 400 \
       --eval-iters 20 \
       --fp16 \
       --adam-eps 1e-6 \
       --optimizer mylamb
