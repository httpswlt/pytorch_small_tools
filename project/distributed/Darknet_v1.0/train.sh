export CUDA_VISIBLE_DEVICES="0,1"
python distribute_train.py  \
--data /home/lingc1/data/ImageNet2012 \
--batch-size 32 \
--lr 0.1 \
--backend nccl \
--url tcp://172.16.123.110:10001 \
--rank 0 \
--world-size 2 \
--last-node-gpus 0 \
--distribute \
-p 1


