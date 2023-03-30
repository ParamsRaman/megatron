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
VOCAB_FILE=~/data/bert-large-uncased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size $4 \
       --pipeline-model-parallel-size 1 \
       --num-layers 6 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --kv-channels 64 \
       --ffn-hidden-size 2048 \
       --encoder-seq-length 512 \
       --decoder-seq-length 128 \
       --micro-batch-size 64 \
       --global-batch-size 2048 \
       --max-position-embeddings 512 \
       --train-iters 25600 \
       --lr-decay-iters 25600 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 5.66e-3 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .04 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 20 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16 \
       --vocab-extra-ids 100 \
       --optimizer lambtp
