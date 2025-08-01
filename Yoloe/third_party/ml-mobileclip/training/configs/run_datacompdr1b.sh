num_gpus=8
num_nodes=16
global_batch_size=$((2**13*8))
num_seen_samples=$((200*1000*global_batch_size))
exp_name="mobileclipb_datacompdr1b_s13b_$(date +%Y-%m-%d_%H-%M-%S)"
num_checkpoints=100  # An epoch is ns/num_checkpoints long
data="DataCompDR-1B/{{00..64}/{00000000..00000999},65/{00000000..00000535}}.tar"

torchrun --nproc_per_node $num_gpus --nnodes $num_nodes --node_rank $ROLE_RANK \
    --max_restarts=0 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_conf "timeout=3000,read_timeout=10000" \
    -m src.training.main \
    --save-frequency 1 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --train-data "$data" \
    --train-num-samples $((num_seen_samples / num_checkpoints)) \
    --warmup 1000 \
    --dataset-type webdataset \
    --precision amp \
    --workers 4 \
    --model ViT-B-16 \
    --batch-size $((global_batch_size / num_nodes / num_gpus)) \
    --epochs $num_checkpoints \
    --lr 1.e-3 \
    --name $exp_name \
    --seed 0 \
    --accum-freq 1 \
    --log-every-n-steps 20 \
    --beta2 0.95 \
    --wd 0.2 \
    --dataset-resampled \
    --resume latest \
    --grad-clip-norm 1.0 \
    --imagenet-val "./imagenet_validation" \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --wandb-project-name mobileclip \
    --dataset-reinforcement \
    --dataset-reinforcement-config configs/datacompdr1b.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.25 0.75 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax
    # For MobileCLIP-S* variants we use the following flag
    # --distill-loss-weights 0.0 1.0
sleep 600
