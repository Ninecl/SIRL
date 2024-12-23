#!/bin/bash
#SBATCH -o %j
#SBATCH -J Grail_WN-SE
#SBATCH -t 72:00:00		#指定作业最大运行30分钟
#SBATCH --mem=90G		#占用节点全部内存
#SBATCH -N 1			#指定节点数
#SBATCH --ntasks-per-node=1	#指定每个节点的进程数
#SBATCH --cpus-per-task=8	#指定每个进程的CPU数，对应多线程场景
#SBATCH --gres=gpu:1		#指定每个节点使用4块GPU卡

python train.py -d WN-SE -e Grail_WN-SE -mt Grail --hop 2 --batch_size 128
python test.py -d WN-SE -e Grail_WN-SE
