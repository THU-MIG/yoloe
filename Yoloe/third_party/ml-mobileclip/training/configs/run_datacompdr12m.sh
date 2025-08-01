num_gpus=8
num_nodes=4
global_batch_size=8192
num_seen_samples=$((30*1000*global_batch_size))
exp_name="mobileclipb_datacompdr12m_s30m_$(date +%Y-%m-%d_%H-%M-%S)"
num_checkpoints=20  # An epoch is ns/num_checkpoints long
data="DataCompDR-12M/shards/{00000000..00001023}.tar"

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
    --save-most-recent \
    --grad-clip-norm 1.0 \
    --imagenet-val "./imagenet_validation" \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --wandb-project-name mobileclip \
    --dataset-reinforcement \
    --dataset-reinforcement-config configs/datacompdr12m.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.0 1.0 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax
sleep 600
