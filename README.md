```
#!/bin/bash
export HF_TOKEN="hf_xxxxxxxxxxxx"
python train.py --outdir=/oscar/data/jtompki1/yhuan170/post_nips_optfull --data=/oscar/data/jtompki1/yhuan170/img_lazy_cfgD/datasets/img256-flux2-mmap --eval=/oscar/data/jtompki1/yhuan170/img_lazy_cfgD/datasets/img256.zip --gpus=8 --batch=4096 --mirror=0 --aug=1 --cond=1 --preset=ImageNet-Ablation --tick=10 --snap=400 --ema-snap=400 --workers=4 --resume=./network-snapshot-000103223.pkl
```