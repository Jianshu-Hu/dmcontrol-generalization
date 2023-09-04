#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=reacher_hard_D1_equi_C2_inv_esvea_export_31250
seed=2

echo "start running $tag with seed $seed"
python src/train.py --algorithm esvea --equi_group D1 --inv_group C2 --export_timesteps 31250 --domain_name reacher --task_name hard --action_repeat 4 --tag $tag --seed $seed
