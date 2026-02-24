```
#!/bin/bash
export HF_TOKEN="hf_xxxxxxxxxxxx"
conda run -n venv2 python train.py --outdir=./training-runs --data=./datasets/img256-flux2-mirror.zip --eval=./datasets/img256.zip --gpus=8 --batch=4096 --mirror=0 --aug=1 --cond=1 --preset=ImageNet-Ablation --tick=10 --snap=400 --g-batch-gpu=256 --d-batch-gpu=256
```