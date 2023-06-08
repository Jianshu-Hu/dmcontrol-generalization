#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=ballincup_catch_svea_random_overlay
seed=1

echo "start running $tag with seed $seed"
python src/train.py --algorithm svea --complex_DA random_overlay --domain_name ball_in_cup --task_name catch --action_repeat 4 --tag $tag --seed $seed
