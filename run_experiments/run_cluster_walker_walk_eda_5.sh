#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=walker_walk_random_crop_mix
seed=5

echo "start running $tag with seed $seed"
python src/train.py --algorithm eda --complex_DA cut_mix --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
