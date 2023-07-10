#!/bin/bash

dataset=$1 # "agg"
gpu=$2 # "0"
model=$3 # "0"
dropout=$4 # "0.5"
nhidden=$5 # "200"
lr=$6 # "1e-3"
epochs=$7 # "2"
batch=$8 # "500"
gamma=$9 # "1"
retrain=${10} # "0"

stime=`date +%s`
dirmodel="models"
dir=$(ls $dirmodel)
for file in ${dir}
do
  echo $file
  cd $dirmodel/$file
  echo "python train.py -d ../../data_local/$dataset --dataset $dataset --gpu $gpu --model $model --dropout $dropout --n-hidden $nhidden --lr $lr --max-epochs $epochs --batch-size $batch --gamma $gamma --retrain $retrain"
  echo "python validate.py -d ../../data_local/$dataset --dataset $dataset --gpu $gpu --model $model --dropout $dropout --n-hidden $nhidden --batch-size $batch --gamma $gamma"
  echo "python test.py -d ../../data_local/$dataset --dataset $dataset --gpu $gpu --model $model --dropout $dropout --n-hidden $nhidden --batch-size $batch --gamma $gamma --epoch=1"
  cd ../..
done


etime=`date +%s`

let ctime=etime-stime

echo $ctime
