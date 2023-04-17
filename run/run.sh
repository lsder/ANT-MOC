#########################################################################
# File Name: run.sh
# Author: lsder
# mail: supreme@lsder.cn
# Created Time: Mon 23 Aug 2021 12:50:10 PM CST
#########################################################################
#!/bin/bash
# sh slurm.job 8 2 2 2
# sh slurm.job 16 2 2 4
# sh slurm.job 16 4 2 2
# sh slurm.job 32 2 4 4
# sh slurm.job 64 4 4 4
sh slurm.job 128 4 8 4 1 4 1 6
sh slurm.job 128 4 8 4 1 8 1 6
sh slurm.job 128 4 8 4 1 16 1 6
sh slurm.job 128 4 8 4 0.1 4 1 6
sh slurm.job 128 4 8 4 0.1 4 0.1 6
# sh slurm.job 128 4 8 4 1 4 1 6
# sh slurm.job 128 4 8 4 1 4 1 6
# sh slurm.job 128 4 8 4 1 4 1 6
# sh slurm.job 256 4 8 8
# sh slurm.job 512 8 8 8
# sh slurm.job 512 64 1 8
# sh slurm.job 128 128 1 1

