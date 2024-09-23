#!/bin/bash
#PBS -N MultiomeModel                  ## job name
#PBS -l nodes=1:ppn=8                 ## 1 node, 8 cores
#PBS -l walltime=48:00:00             ## max. 48h of wall time
#PBS -l mem=64gb                      ## 64 Gb memory

module load IPython/7.26.0-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
module load matplotlib/3.4.3-foss-2021b
module load scikit-learn/1.0.1-foss-2021b
module load scikit-image/0.19.1-foss-2021b
module load scanpy/1.9.1-foss-2021b

echo "Module Loading Complete"

cd data/DesignProject/classification/scripts

python3 -u BMMC_multiome_svm.py >> BMMC_multiome_svm.log 2>&1
cat BMMC_multiome_svm.log
