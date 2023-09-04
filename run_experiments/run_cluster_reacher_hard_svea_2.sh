#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=reacher_hard_svea
seed=2

echo "start running $tag with seed $seed"
python src/train.py --algorithm svea --complex_DA random_overlay --domain_name reacher --task_name hard --action_repeat 4 --tag $tag --seed $seed
