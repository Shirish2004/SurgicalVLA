policy_name=SurgicalVLA
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONWARNINGS=ignore::UserWarning

cd ../..
uv run scripts/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    ${CKPT:+--checkpoint_path "$CKPT"}
