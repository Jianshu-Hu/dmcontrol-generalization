#!/bin/bash

cd /bigdata/users/jhu/dmcontrol-generalization
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=walker_walk_cro_cut_once
seed=5

echo "start running $tag with seed $seed"
python src/train.py --algorithm cro --complex_DA random_overlay --domain_name walker --task_name walk --action_repeat 4 --tag $tag --seed $seed
