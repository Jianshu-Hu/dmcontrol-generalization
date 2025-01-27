#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_jianshu

tag=cartpole_swingup_weighted_cut_random_overlay_q_diff_kl_diff_AC_proj_grad
seed=4

echo "start running $tag with seed $seed"
python src/train.py --algorithm wro --wro_weight_type q_diff_kl_diff --projected_grad --complex_DA cut_random_overlay --domain_name cartpole --task_name swingup --action_repeat 8 --tag $tag --seed $seed
