#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=walker_walk_weighted_cut_random_overlay_q_diff_kl_diff_AC_proj_grad
seed=2

echo "start running $tag with seed $seed"
python src/train.py --algorithm wro --wro_weight_type q_diff_kl_diff --projected_grad --complex_DA cut_random_overlay --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
