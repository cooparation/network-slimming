VOC07_PATH=/datasets/VOCdevkit/VOC2007
VOC12_PATH=/datasets/VOCdevkit/VOC2012
SAVE_PATH=./models/mbv2_ssdlite_prune_finetune/
    #--pretrained_ssd $PRETRAINED_MODEL \
    #--base_net models/mb2-imagenet-71_8.pth \
    #--scheduler cosine --lr 0.01 --t_max 200 \
PRETRAINED_MODEL='pruned.pth'
python train_ssd.py --dataset_type voc \
    --datasets $VOC07_PATH $VOC12_PATH \
    --validation_dataset $VOC07_PATH \
    --checkpoint_folder $SAVE_PATH \
    --net mb2-ssd-lite-prune \
    --pretrained_ssd $PRETRAINED_MODEL \
    --scheduler cosine --lr 0.01 --t_max 400 \
    --validation_epochs 5 --num_epochs 400 \
    --base_net_lr 0.01 --extra_layers_lr 0.01
