#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=cartpole_swingup_svea_random_overlay
seed=4

echo "start running $tag with seed $seed"
python src/train.py --algorithm svea --complex_DA random_overlay --domain_name cartpole --task_name swingup --action_repeat 8 --tag $tag --seed $seed
