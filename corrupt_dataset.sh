dataset="coco"
dir2save="./data/corruptions/datasets/coco"
limit=1000

python3 scripts/corruptions.py \
    --dataset=$dataset \
    --dir2save=$dir2save \
    --limit=$limit