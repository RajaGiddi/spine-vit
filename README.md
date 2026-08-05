# Spine-ViT

Grades spinal canal stenosis on lumbar spine MRI, one disc level at a time. It pools one
token per vertebral level out of a frozen DINOv2 backbone, so every prediction belongs to a
named level.

## Install

Use Python 3.11 or 3.12. PyTorch has no wheels for 3.14 yet.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Not included. Download the RSNA 2024 Lumbar Spine Degenerative Classification set from

https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

and set `RSNA_DIR` to its path.

## Run

```
python rsna_setup.py --mode subset --n 500 --out_dir $RSNA_DIR
python scripts/explore_rsna.py --data_dir $RSNA_DIR
python train.py --data_dir $RSNA_DIR --dataset rsna --tokenizer anatomy --pos_encoding ordinal
python evaluate.py --experiments_dir outputs --data_dir $RSNA_DIR --generate_figures
```

## Test

These run without any data.

```
python tests/test_forward.py
python tests/test_train_loop.py
```
