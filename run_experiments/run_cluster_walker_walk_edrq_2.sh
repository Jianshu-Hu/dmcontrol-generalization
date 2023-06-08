#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=walker_walk_edrq_shift
seed=2

echo "start running $tag with seed $seed"
python src/train.py --algorithm edrq --data_aug_type shift --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
