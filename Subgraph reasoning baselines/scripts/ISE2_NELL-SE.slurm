#!/bin/bash
#SBATCH -o %j
#SBATCH -J ISE2_NELL-SE
#SBATCH -t 72:00:00		#指定作业最大运行30分钟
#SBATCH --mem=90G		#占用节点全部内存
#SBATCH -N 1			#指定节点数
#SBATCH --ntasks-per-node=1	#指定每个节点的进程数
#SBATCH --cpus-per-task=6	#指定每个进程的CPU数，对应多线程场景
#SBATCH --gres=gpu:1		#指定每个节点使用4块GPU卡

h=2 # hop
bs=512  # batch size
nw=6   # number workers
neg=1   # number of neg samples
lr=0.001    # learning rate
mh=0  #  max nodes per hop
n_RT=0 # number of RTransformer layers
GP_m=sGP
RT_m=sRT

if [ ${mh} -gt 0 ]; then
    en=ISE2_NELL-SE_${GP_m}_${RT_m}${n_RT}_hop${h}_max${mh}_neg${neg}
    python train.py -d NELL-SE -e ${en} --hop ${h} -bs ${bs} -nw ${nw} -neg ${neg} --lr ${lr} -max_h ${mh} -n_RT ${n_RT} -GP_m ${GP_m} -RT_m ${RT_m}
else
    en=ISE2_NELL-SE_${GP_m}_${RT_m}${n_RT}_hop${h}_all_neg${neg}
    python train.py -d NELL-SE -e ${en} --hop ${h} -bs ${bs} -nw ${nw} -neg ${neg} --lr ${lr} -n_RT ${n_RT} -GP_m ${GP_m} -RT_m ${RT_m}
fi

python test.py -d NELL-SE -e ${en} --num_workers ${nw} -ss