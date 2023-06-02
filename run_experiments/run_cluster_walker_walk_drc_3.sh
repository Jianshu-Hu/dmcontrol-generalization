#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=walker_walk_drc_rerun_categorical_detach_encoder
seed=3

echo "start running $tag with seed $seed"
python src/train.py --algorithm drc --drc_dist_type categorical --complex_DA random_conv --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
