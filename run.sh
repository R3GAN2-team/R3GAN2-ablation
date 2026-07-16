#!/bin/bash
eval "$(/users/jwoodlei/miniconda3/bin/conda shell.bash hook)"
conda activate sm100

unset LD_LIBRARY_PATH

export CUTLASS_DIR="$HOME/cutlass-main"
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxx"
python train.py --outdir=/oscar/data/jtompki1/yhuan170/post_nips_resume_2x --data=/oscar/data/jtompki1/yhuan170/img_lazy_cfgD/datasets/img256-flux2-mmap --eval=/oscar/data/jtompki1/yhuan170/img_lazy_cfgD/datasets/img256.zip --gpus=8 --batch=4096 --mirror=0 --aug=1 --cond=1 --preset=ImageNet-2x --tick=10 --snap=400 --ema-snap=400 --workers=4 --resume=1
