#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=reacher_hard_drq_shift
seed=3

echo "start running $tag with seed $seed"
python src/train.py --algorithm drq --data_aug_type shift --domain_name reacher --task_name hard --action_repeat 4 --tag $tag --seed $seed
