##### This script run RAG-QA Arena by compare with LFRQA directly ######

# A few examples of model options,
# model1=meta-llama/Meta-Llama-3-70B-Instruct
# model1=mistralai/Mixtral-8x22B-Instruct-v0.1
# model1=CohereForAI/c4ai-command-r-plus
# model1=Qwen/Qwen1.5-32B-Chat
# model1=gpt-4-turbo
# model1=gpt-4o
model1=internlm2_5-7b-chat
# model1=gpt-4
# model1=gpt-4o-mini

openai_key='Your api key'

# eval_model=gpt-4-0125-preview
# eval_model=gpt-4o-mini
eval_model=Llama-3-70B-Instruct
eval_model_url=http://22.8.22.131:23333

n_passages=1
domains=(lifestyle recreation technology science writing)
for i in "${!domains[@]}"
do
    echo evaluating ${model1} against LFRQA using ${eval_model} for ${domains[i]}
    python code/evaluate_pair_responses.py \
        --eval_dir "data/pairwise_eval" \
        --model ${eval_model} \
        --model_url ${eval_model_url} \
        --eval_model1 ${model1} \
        --model1_pred_file  ${model1}/${domains[i]}_from_colbert_use_gt_ctx${n_passages}_psgs  \
        --reference_file references \
        --template_config pairwise_lfrqa.cfg \
        --domain ${domains[i]} \
        --temperature 0.0 \
        --inference_batch_size 16 \
        --eval_input_save_dir eval_inputs/from_colbert/use_gt_ctx \
        --api_key ${openai_key}
done