features_path="./data/corruptions/features/corrupted_coco_features.pickle"
output_dir="./results/retrieved_images/"
heatmap_dir="./results/heatmaps/"
dataset_path="./data/corruptions/datasets/coco/"
k_value=3
device="cuda"

python3 scripts/retrieve_clip.py \
    --features-path=$features_path \
    --output-dir=$output_dir \
    --dataset-path=$dataset_path \
    --heatmap-dir=$heatmap_dir \
    --k-value=$k_value \
    --device=$device
