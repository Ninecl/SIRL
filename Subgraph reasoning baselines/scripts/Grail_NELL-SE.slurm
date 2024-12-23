#!/bin/bash

h=1 # hop
bs=512  # batch size
nw=12   # number workers
neg=1   # number of neg samples
lr=0.001    # learning rate
mh=0  #  max nodes per hop
n_RT=1 # number of RTransformer layers
mt=Grail

#SBATCH -o %j
#SBATCH -J Grail_FB-SE
#SBATCH -t 72:00:00		#指定作业最大运行30分钟
#SBATCH --mem=90G		#占用节点全部内存
#SBATCH -N 1			#指定节点数
#SBATCH --ntasks-per-node=1	#指定每个节点的进程数
#SBATCH --cpus-per-task=12	#指定每个进程的CPU数，对应多线程场景
#SBATCH --gres=gpu:1		#指定每个节点使用4块GPU卡

if [ ${mh} -gt 0 ]; then
    en=${mt}_FB-SE_GP_RT${n_RT}_hop${h}_max${mh}_neg${neg}
    python train.py -d FB-SE -e ${en} --hop ${h} -bs ${bs} -nw ${nw} -neg ${neg} --lr ${lr} -max_h ${mh} -n_RT ${n_RT} -mt ${mt}
else
    en=${mt}_FB-SE_GP_RT${n_RT}_hop${h}_all_neg${neg}
    python train.py -d FB-SE -e ${en} --hop ${h} -bs ${bs} -nw ${nw} -neg ${neg} --lr ${lr} -n_RT ${n_RT} -mt ${mt}
fi

python test.py -d FB-SE -e ${en} --num_workers ${nw} -ss