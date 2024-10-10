##### Script to run HF models #####

# WARNING: you will need to specify clm_new_token_trigger for different models!!
# Here is the mapping
# mistralai/Mixtral*:          "[/INST]"
# meta-llama/Meta-Llama*:      "<|start_header_id|>assistant<|end_header_id|>"
# CohereForAI/c4ai-command-r*: "<|CHATBOT_TOKEN|>"
# Qwen/Qwen1.5*:               "<|im_start|>assistant"


# model_path="meta-llama/Meta-Llama-3-70B-Instruct"
# model_path="CohereForAI/c4ai-command-r-v01"
# model_path="Qwen/Qwen1.5-32B-Chat"
# model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_path=/cpfs02/llm/shared/public/zhaoqian/ckpt/7B/240623/P-volc_internlm2_5_boost1_7B_FT_merge_boost_bbh_v2
# model_name=internlm2_5-7b-chat
model_name=gpt-4o-mini
model_path='api'

if [[ ${model_name} == gpt-4o* ]]; then
    template_version=v2
else
    template_version=v1
fi

openai_key='Your api key'

# domains=(bioasq fiqa lifestyle recreation technology science writing)
# --inject_negative_ctx true \
domains=(lifestyle recreation technology science writing)
for i in "${!domains[@]}"; do
    devices="0"
    echo eval ${domains[i]} on ${devices} using ${model_name} template ${template_version}
    export CUDA_VISIBLE_DEVICES=${devices}
    python code/generate_responses.py \
        --model ${model_name} \
        --model_path ${model_path} \
        --tp 1 \
        --input_file /cpfs02/llm/shared/public/liujiangning/work/MindSearchEval/robustqa-acl23/output/${domains[i]}_from_colbert_test.jsonl \
        --output_file ${domains[i]}_from_colbert \
        --template_config ans_generation_${template_version}.cfg \
        --domain ${domains[i]} \
        --use_gt_ctx true \
        --n_passages 5 \
        --inference_batch_size 16 \
        --eval_dir "data/pairwise_eval/${model_name}" \
        --api_key ${openai_key}
done
