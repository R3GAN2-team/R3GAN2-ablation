#!/bin/bash
eval "$(/users/yhuan170/anaconda3/bin/conda shell.bash hook)"

unset LD_LIBRARY_PATH

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python async_eval.py \
  --run-dir /oscar/data/jtompki1/yhuan170/post_nips_resume_2x/00000-img256-flux2-mmap-gpus8-batch4096 \
  --gpus 2 \
  --ema-stds 0.100 
