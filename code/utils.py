import logging
import sys
import re
from datasets import Dataset

loggers = {}

def get_logger(name='default'):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger
    
def chunk(iterable, chunksize):
    # if iterable is a list, we chunk with simple list indexing
    if isinstance(iterable, list):
        return [iterable[i:i+chunksize] for i in range(0, len(iterable), chunksize)]
    # otherwise if iterable is a Hf Dataset, we leverage the select() function to create mini datasets
    elif isinstance(iterable, Dataset):
        chunks = []
        for i in range(0, len(iterable), chunksize):
            if i+chunksize < len(iterable):
                chunks.append(iterable.select(list(range(i, i+chunksize))))
            else:
                chunks.append(iterable.select(list(range(i, len(iterable)))))
        return chunks
    else:
        raise Exception(f"Unrecognizable type of iterable for batchification: {type(iterable)}")
    
def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens. See model_args.clm_new_token_trigger
    # if cannot find trigger --> generation is too long; default to empty generation
    # we use rfind to find the last occurence of the pattern in sequence; this is compatible with ICL evaluation
    start = pred.rfind(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].strip() # strip any side whitespaces
    return output

def extract_prompt(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens. See model_args.clm_new_token_trigger
    # if cannot find trigger --> generation is too long; default to empty generation
    # we use rfind to find the last occurence of the pattern in sequence; this is compatible with ICL evaluation
    start = pred.rfind(trigger)
    if start < 0:
        return ''
    prompt = pred[:start].strip() # strip any side whitespaces
    return prompt

def remove_elements(response, open='<thinking>', close='</thinking>'):
    '''this works because response by assuming 1 tag per response'''
    start_idx = response.find(open)
    close_idx = response.find(close)

    if start_idx >= 0 and close_idx >= 0 and start_idx <= close_idx:
        # remove the thinking substring
        response = response[close_idx+len(close):]
    return response.strip()

def replace_answer_tags(response, open_bracket='<answer>', close_bracket='</answer>'):
    return response.replace(open_bracket, '').replace(close_bracket, '')

def extract_answer(response, open_bracket='<answer>', close_bracket='</answer>'):
    extracted = re.findall(
        rf"{open_bracket}(.*?){close_bracket}",
        response,
        re.DOTALL)
    if extracted:
        return extracted[0]
    return response.replace(open_bracket, '').replace(close_bracket, '')

def replace_null_answer(response):
    ans_patterns = ['FAIL TO GENERATE ANS.']
    if response in ans_patterns:
        return "I couldn't find an answer."
    return response

def process_response(response):
    response = remove_elements(response)
    response = remove_elements(response, 'Thinking: ', 'Answer: ')
    response = remove_elements(response, 'Thoughts: ', 'Answer: ')
    response = remove_elements(response, 'Thought: ', 'Answer: ')
    # response = replace_answer_tags(response, 'Answer: ')
    # response = replace_answer_tags(response, 'Answer:\n')
    # response = replace_answer_tags(response)
    response = re.split(r'<passage[^>]*>', response)[-1]
    response = re.split(r'</passage[^>]*>', response)[-1]
    response = response.split('<query>')[-1]
    response = response.split('</query>')[-1]
    response = response.split('Answer: ')[-1]
    response = response.split('<answer>')[-1]
    response = replace_answer_tags(response)
    response = replace_null_answer(response)
    return response

def is_hf_model(model_name):
    hf_models = ['mistral', 'cohere']
    if any([m in model_name.lower() for m in hf_models]):
        return True
    else:
        return False

def group_data_by_task(input_file='data/human_eval/LFRQA_gpt-4_eval_by_gpt-4-0125-preview.json', save_to='eval_inputs/from_colbert'):
    import json
    import os
    with open(input_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    grouped = dict()
    for item in data:
        pair_id = item['pair_id'].split('-')[0]
        if pair_id not in grouped:
            grouped[pair_id] = [item]
        else:
            grouped[pair_id].append(item)
    for task_name, data_points in grouped.items():
        print(f'num of data points for task {task_name}: {len(data_points)}')
        with open(os.path.join(save_to, f'{task_name}_gpt-4.json'), 'w', encoding='utf-8') as fw:
            json.dump(data_points, fw, ensure_ascii=False)


if __name__ == '__main__':
    group_data_by_task()