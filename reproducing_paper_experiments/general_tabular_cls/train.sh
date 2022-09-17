# Supervised learning case


declare -a StringArray=( '1995_income' )

for value in "${StringArray[@]}";
do

declare -a Models=( 'FiFa' )
for model in "${Models[@]}";

do


feat_type="all"
embedding_size=32

declare -a twoLayersDim=(   2  )
for final_dim_factor in "${twoLayersDim[@]}";

do

declare -a trainMaskProb=( 0.0 )
for train_mask_prob in "${trainMaskProb[@]}";

do

declare -a twoLayersMlpDimFactor=( 3 )
for num_dim_factor in "${twoLayersMlpDimFactor[@]}";

do

declare -a twoLayersDropout=( 0.5 )
for dropout in "${twoLayersDropout[@]}";

do


declare -a twoLayersIncludeY=( 0 )
for include_y in "${twoLayersIncludeY[@]}";

do



declare -a Act=( 'quad'  )
for act in "${Act[@]}";

do


python train.py --dataset $value  --project_name "fifa" --run_name nopt_${model}_${feat_type}_${value}_dim-factor-${final_dim_factor}_num-dim-factor-${num_dim_factor}_mf-dropout-${dropout}_include-y-${include_y}_emb-size-${embedding_size}}_act-${act} --group_name nopt_${feat_type}_${value}  --model $model --feat_type ${feat_type} --epochs 100 --embedding_size ${embedding_size} --batchsize 256 --include_y ${include_y} --dropout ${dropout} --num_dim_factor ${num_dim_factor} --train_mask_prob ${train_mask_prob} --final_dim_factor ${final_dim_factor}  --act ${act}


done
done
done
done
done
done
done
done