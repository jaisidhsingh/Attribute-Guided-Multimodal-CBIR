input_dir="./data/coco/train2014/"
output_name="coco_folder"
output_dir="./data/coco/"

python3 scripts/get_clip_features.py \
    --input-dir=$input_dir \
    --output-name=$output_name \
    --output-dir=$output_dir
