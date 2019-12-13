VOC07_PATH=/datasets/VOCdevkit/VOC2007
VOC12_PATH=/datasets/VOCdevkit/VOC2012
SAVE_PATH=./models/mbv2_ssdlite_sparse/
PRETRAINED_MODEL='models/mbv2_ssdlite_prune/mb2-ssd-lite-prune-0.8047.pth'
    #--base_net models/mb2-imagenet-71_8.pth \
    #--scheduler cosine --lr 0.1 --t_max 300 \
    #--base_net_lr 0.1 --extra_layers_lr 0.01
    #--scheduler multi-step --lr 0.1 --milestones '80,100,150,200,280'\
python train_ssd.py --dataset_type voc \
    --sparsity-regularization --s 0.001 \
    --datasets $VOC07_PATH $VOC12_PATH \
    --validation_dataset $VOC07_PATH \
    --checkpoint_folder $SAVE_PATH \
    --net mb2-ssd-lite-prune \
    --scheduler cosine --lr 0.01 --t_max 200 \
    --pretrained_ssd $PRETRAINED_MODEL \
    --validation_epochs 5 --num_epochs 200 \
    --base_net_lr 0.01 --extra_layers_lr 0.01
