#!/bin/sh
envID="2d_maze-v11"
model="DT" # "RvS" "DT"
max_epochs_ql=2500
max_epochs_dt=1500 
post_fix="_100"
cql_gamma=0.99
qdt_gamma=1.00

# CAL & QDT simulation
replace=1
for n in 1 2 3 4 5 6 7 8 9 10
do
    python train_cql_dt-model-v1.py --envID $envID --max_epochs_ql $max_epochs_ql --max_epochs_dt $max_epochs_dt --post_fix $post_fix --model $model --cql_gamma $cql_gamma --qdt_gamma $qdt_gamma --replace $replace
    python eval_cql_dt-model-v1.py --envID $envID --max_epochs_ql $max_epochs_ql --max_epochs_dt $max_epochs_dt --post_fix $post_fix --model $model --qdt_gamma $qdt_gamma
done

# DT simulation
max_epochs_ql=500 
replace=0
for n in 1 2 3 4 5 6 7 8 9 10
do
    python train_cql_dt-model-v1.py --envID $envID --max_epochs_ql $max_epochs_ql --max_epochs_dt $max_epochs_dt --post_fix $post_fix --model $model --replace $replace
    python eval_cql_dt-model-v1.py --envID $envID --max_epochs_ql $max_epochs_ql --max_epochs_dt $max_epochs_dt --post_fix $post_fix --model $model
done

