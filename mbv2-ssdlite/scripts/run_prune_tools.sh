MODEL_PATH='models/mbv2_ssdlite_sparse_008/mb2-ssd-lite-prune-Epoch-345-Loss-2.015319828745685.pth'
python prune_mbv2ssdlite_tools.py --net mb2-ssd-lite-prune \
    --model_path $MODEL_PATH \
    --prune_percent 0.5 \
    --label_file models/voc-model-labels.txt
