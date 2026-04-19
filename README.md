```
#!/bin/bash
conda run -n venv2 python train.py --outdir=./training-runs --data=./datasets/cifar10.zip --gpus=8 --batch=512 --mirror=1 --aug=1 --cond=1 --preset=CIFAR10 --tick=10 --snap=20 --ema-snap=20
```