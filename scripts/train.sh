#!/bin/bash 
cd src

dataset=$1
similarity=$2
is_hyperbolic=$3
prior_mode=$4
lr=$5
total_epoch=$6
num_hyp_layers=$7
hyp_hidden_dim=$8
hyp_output_dim=$9
num_fc_layers=${10}
fc_hidden_dim=${11}
triplet_selector=${12}
ssl_coeff=${13}
margin=${14}
reg_type=${15}
reg_coeff=${16}
reg_margin=${17}

echo "python train.py --dataset ${dataset} --similarity ${similarity} --is_hyperbolic ${is_hyperbolic} --prior_mode ${prior_mode} --lr ${lr} --total_epoch ${total_epoch} --num_hyp_layers ${num_hyp_layers} --hyp_hidden_dim ${hyp_hidden_dim} --hyp_output_dim ${hyp_output_dim} --num_fc_layers ${num_fc_layers} --fc_hidden_dim ${fc_hidden_dim} --triplet_selector ${triplet_selector} --ssl_coeff ${ssl_coeff} --margin ${margin} --reg_type ${reg_type} --reg_coeff ${reg_coeff} --reg_margin ${reg_margin}"
python train.py --dataset ${dataset} --similarity ${similarity} --is_hyperbolic ${is_hyperbolic} --prior_mode ${prior_mode} --lr ${lr} --total_epoch ${total_epoch} --num_hyp_layers ${num_hyp_layers} --hyp_hidden_dim ${hyp_hidden_dim} --hyp_output_dim ${hyp_output_dim} --num_fc_layers ${num_fc_layers} --fc_hidden_dim ${fc_hidden_dim} --triplet_selector ${triplet_selector} --ssl_coeff ${ssl_coeff} --margin ${margin} --reg_type ${reg_type} --reg_coeff ${reg_coeff} --reg_margin ${reg_margin}
