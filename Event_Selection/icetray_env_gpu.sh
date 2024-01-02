#!/bin/bash

echo $HOSTNAME
echo $PATH
echo $PYTHONPATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export KERAS_BACKEND="tensorflow"
export HDF5_USE_FILE_LOCKING=FALSE
if [ ! -e /usr/local/cuda/bin/ ]; then
    echo "Running on CPU!"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/
else
   echo "Running on GPU!"
fi
export TMPDIR=/data/user/xk35/docker2/temp/
export SINGULARITY_TMPDIR=/data/user/xk35/docker2/temp/
export SINGULARITY_CACHEDIR=/data/user/xk35/docker2/cache/
export DNN_BASE=/home/tglauch/I3Module/
export pythonpath=/usr/local/lib/
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$PATH:${CUDA_HOME}/bin:${CUDA_HOME}/include
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/lib/
singularity exec --nv -B /home/$USER/:/home/$USER/ -B /data/user/:/data/user/ -B /data/ana/:/data/ana/ -B /data/sim/:/data/sim/ /data/user/xk35/docker2/icetray_combo-stable-tensorflow2.4.1-ubuntu20.04.sif /data/user/xk35/software/meta_projects/singularity_v01-01-00/build/env-shell.sh python3 $@


