
# RAN

## Requirements

+ torch==1.3.0+
+ numba

- Training

    ```shell
    python train.py \
    --checkpoint enfr-base \
    --model transformer \
    --langs en fr \ # language suffixes, optional
    --train-bin $DATA/bin \
    --train train \ # auto-expands the arguments if `--langs` available
    --vocab vocab.join \ # auto-expands the arguments if `--langs` available
    --dev $DATA/test13 \ # auto-expands the arguments if `--langs` available
    --max-step 100000 \
    --att-enc-type recrand \
    --att-dec-type recrand \
    --max-epoch 0 \
    --lr 7e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 \
    --warmup-steps 4000 \
    --optimizer adam \
    --num-workers 6 \
    --max-tokens 8333 \ # effective batch size is (max_tokens * dist_world_size * accumulate)
    --dist-world-size 3 \ # number of available GPUs
    --accumulate 1 \
    --seed 9527 \ 
    --save-checkpoint-steps 2000 \
    --save-checkpoint-secs 0 \
    --save-checkpoint-epochs 1 \
    --keep-checkpoint-max 10 \
    --keep-best-checkpoint-max 2 \
    --shuffle 1 \
    --input-size 512 \
    --hidden-size 512 \
    --ffn-hidden-size 2048 \
    --num-heads 8 \
    --share-all-embedding 1 \
    --residual-dropout 0.1 \
    --attention-dropout 0 \
    --ffn-dropout 0 \
    --val-method bleu \ # available validation method: bleu/logp
    --val-steps 1000 \ # validation frequncey
    --val-max-tokens 4096 \ # validation batch-size
    --fp16 half \ # mixed-precision training: none/half/amp
    > log.train 2>&1 
    ```
    
    Type `python train.py -h` for more available options for model, optimizer, lr_scheduler, dataset, etc.

    