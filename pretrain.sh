cd /data/ydchen/VLP/bigmodel/IJCAI23/MAE && \
python -m torch.distributed.launch --nproc_per_node=4 train_MAE.py \