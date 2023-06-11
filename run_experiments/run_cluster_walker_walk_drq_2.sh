#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=walker_walk_drq_rot_90
seed=2

echo "start running $tag with seed $seed"
python src/train.py --algorithm drq --data_aug_type rot --degrees 90 --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
