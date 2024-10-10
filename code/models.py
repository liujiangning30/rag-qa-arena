from typing import List, Union

from openai import OpenAI
import torch
from tqdm import tqdm
from utils import chunk, extract_output, extract_prompt, is_hf_model
import time

from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

class GenericModel():
    def __init__(self, config):
        self.config = config

    def generate(self, data):
        pass

class HFModel(GenericModel):
    def __init__(self, config):
        super().__init__(config)

        if config.load_in_8bit and config.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif config.load_in_8bit or config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config.load_in_8bit, 
                load_in_4bit=config.load_in_4bit
            )
            # This means: fit the entire model on a GPU
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = "auto"
            quantization_config = None
            if config.bf16:
                torch_dtype = torch.bfloat16
            elif config.fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = None

        self.model = AutoModelForCausalLM.from_pretrained(
                    config.model,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=config.trust_remote_code,
                    use_auth_token=config.use_auth_token
            )

    def run_predictions(self, data, config, tokenizer=None):
        res = []
        pad_token_id = tokenizer.eos_token_id    
        for batch_example in tqdm(chunk(data, config.inference_batch_size)):
            input_ids = torch.tensor(batch_example["input_ids"])
            attention_mask = torch.tensor(batch_example["attention_mask"])
            preds = self.model.generate(input_ids=input_ids.to(self.model.device), 
                                        attention_mask=attention_mask.to(self.model.device), 
                                        max_new_tokens=self.config.max_new_tokens,
                                        do_sample=self.config.do_sample,
                                        pad_token_id=pad_token_id)
            # do not skip special tokens as they are useful to split prompt and pred
            decoded_preds = tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=False)
            for pred, example in zip(decoded_preds, batch_example):
                if tokenizer.eos_token and tokenizer.bos_token:
                    pred = pred.replace(tokenizer.eos_token, '').replace(tokenizer.bos_token, '')
                example['prompt'] = extract_prompt(pred, self.config.clm_new_token_trigger)
                example['pred'] = extract_output(pred, self.config.clm_new_token_trigger)
                # remove special tokens after split prompt and pred
                if tokenizer.additional_special_tokens:
                    for special_token in tokenizer.additional_special_tokens:
                        example['pred'] = example['pred'].replace(special_token, '')
                example.pop('input_ids')
                example.pop('attention_mask')
                res.append(example)
        return res


class OpenAIModel(GenericModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = OpenAI(api_key=config.api_key)
        self.model_name = config.model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):
        res = []
        fail_count = 0
        for ex in tqdm(data):
            retry = 0
            # if pairwise comparison
            messages = []
            if 'ans_generation' not in args.template_config:
                messages.append({"role": "system", "content": ex["system"]})
                for turn in ex['examples']:
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
            messages.append({"role": "user", "content": ex['prompt']})
            while retry < 5:
                try:
                    ans = self.model.chat.completions.create(model=self.model_name, 
                                                             messages=messages,
                                                             max_tokens=args.max_new_tokens,
                                                             temperature=args.temperature,
                                                             top_p=args.top_p)
                    ex['pred'] = ans.choices[0].message.content
                    break
                except:
                    retry += 1
                    time.sleep(1)
            if retry == 5:
                fail_count += 1
                ex['pred'] = 'FAIL TO GENERATE ANS.'
                print(f"failed {fail_count} times")
            res.append(ex)
        
        return res

    
class OpenAIModel(GenericModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = OpenAI(api_key=config.api_key)
        self.model_name = config.model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):
        res = []
        fail_count = 0
        for ex in tqdm(data):
            retry = 0
            # if pairwise comparison
            messages = []
            if 'ans_generation' not in args.template_config:
                messages.append({"role": "system", "content": ex["system"]})
                for turn in ex['examples']:
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
            messages.append({"role": "user", "content": ex['prompt']})
            while retry < 5:
                try:
                    ans = self.model.chat.completions.create(model=self.model_name, 
                                                             messages=messages,
                                                             max_tokens=args.max_new_tokens,
                                                             temperature=args.temperature,
                                                             top_p=args.top_p)
                    ex['pred'] = ans.choices[0].message.content
                    break
                except:
                    retry += 1
                    time.sleep(1)
            if retry == 5:
                fail_count += 1
                ex['pred'] = 'FAIL TO GENERATE ANS.'
                print(f"failed {fail_count} times")
            res.append(ex)
        
        return res


class OpenAIModelAPI(GenericModel):
    def __init__(self,
                 config,
                 proxies=dict(
                    http='http://liujiangning:QvNIdAiv3QkiXOB3Kpx24kum6KpEievWYfbu1cPO0FJRqDPU8Zo1nz79bolY@closeai-proxy.pjlab.org.cn:23128',
                    https='http://liujiangning:QvNIdAiv3QkiXOB3Kpx24kum6KpEievWYfbu1cPO0FJRqDPU8Zo1nz79bolY@closeai-proxy.pjlab.org.cn:23128'
                 ),
                 max_new_tokens: int = 512,
                 top_p: float = 0.8,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0):
        super().__init__(config)

        from lagent.llms import GPTAPI
        model = GPTAPI(
            model_type=config.model,
            key=config.api_key,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            proxies=proxies,
            query_per_second=50,
            retry=1000)
        self.model = model
        self.model_name = config.model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):
        batch_indexes = range(0, len(data), args.inference_batch_size)
        res = []
        for spilt_ids in (tqdm(batch_indexes, desc="inferencing")):
            batch_data_points = data[spilt_ids: spilt_ids+args.inference_batch_size]
            batch_messages = []
            for ex in batch_data_points:
                messages = []
                if 'ans_generation' not in args.template_config:
                    messages.append({"role": "system", "content": ex["system"]})
                    for turn in ex['examples']:
                        messages.append({"role": "user", "content": turn['user']})
                        messages.append({"role": "assistant", "content": turn['assistant']})
                messages.append({"role": "user", "content": ex['prompt']})
                batch_messages.append(messages)
            batch_responses = self.model.chat(batch_messages)
            for resp, ex in zip(batch_responses, batch_data_points):
                ex['pred'] = resp
                res.append(ex)
        return res


class LmdeployModel(GenericModel):
    def __init__(self, config,
                 max_new_tokens: int = 1024,
                 top_p: float = 0.8,
                 top_k: float = 1,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 stop_words: List[str] = ['<|im_end|>']):
        super().__init__(config)

        from lagent.llms import LMDeployPipeline, INTERNLM2_META
        model = LMDeployPipeline(
            path=config.model_path,
            model_name=config.model,
            meta_template=INTERNLM2_META,
            tp=config.tp,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words)
        self.model = model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):
        batch_indexes = range(0, len(data), args.inference_batch_size)
        res = []
        for spilt_ids in (tqdm(batch_indexes, desc="inferencing")):
            batch_data_points = data[spilt_ids: spilt_ids+args.inference_batch_size]
            batch_messages = []
            for ex in batch_data_points:
                messages = []
                if 'ans_generation' not in args.template_config:
                    messages.append({"role": "system", "content": ex["system"]})
                    for turn in ex['examples']:
                        messages.append({"role": "user", "content": turn['user']})
                        messages.append({"role": "assistant", "content": turn['assistant']})
                messages.append({"role": "user", "content": ex['prompt']})
                batch_messages.append(messages)
            batch_responses = self.model.chat(batch_messages)
            for resp, ex in zip(batch_responses, batch_data_points):
                ex['pred'] = resp
                res.append(ex)
        return res


class LLama3ModelClient(GenericModel):
    def __init__(self, config,
                 max_new_tokens: int = 1024,
                 top_p: float = 0.8,
                 top_k: float = 1,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 stop_words: List[str] = ['<|eot_id|>']):
        super().__init__(config)

        from lagent.llms import LMDeployClient

        meta_template = [
            dict(
                role='system',
                begin='<|start_header_id|>system<|end_header_id|>\n',
                end='<|eot_id|>',
            ),
            dict(
                role='user',
                begin='<|start_header_id|>user<|end_header_id|>\n',
                end='<|eot_id|>'
            ),
            dict(
                role='assistant',
                begin='<|start_header_id|>assistant<|end_header_id|>',
                end='<|eot_id|>'
            ),
            dict(
                role='environment',
                begin='<|start_header_id|>system<|end_header_id|>\n',
                end='<|eot_id|>'),
        ]
        model = LMDeployClient(
            model_name=config.model,
            url=config.model_url,
            meta_template=meta_template,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words)
        self.model = model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):

        batch_indexes = range(0, len(data), args.inference_batch_size)
        res = []
        for spilt_ids in (tqdm(batch_indexes, desc="inferencing")):
            batch_data_points = data[spilt_ids: spilt_ids+args.inference_batch_size]
            batch_prompts = []
            for ex in batch_data_points:
                messages = []
                if 'ans_generation' not in args.template_config:
                    messages.append({"role": "system", "content": ex["system"]})
                    for turn in ex['examples']:
                        messages.append({"role": "user", "content": turn['user']})
                        messages.append({"role": "assistant", "content": turn['assistant']})
                messages.append({"role": "user", "content": ex['prompt']})
                prompt = '<|begin_of_text|>' + self.model.template_parser(messages)
                batch_prompts.append(prompt)
            batch_responses = self.model.generate(batch_prompts)
            for resp, ex in zip(batch_responses, batch_data_points):
                ex['pred'] = resp
                res.append(ex)
        return res


def load_model(args, logger):
    if 'gpt' in args.model:
        model = OpenAIModelAPI(args)
    elif 'llama' in args.model.lower():
        model = LLama3ModelClient(args)
    elif is_hf_model(args.model):
        model = HFModel(args)
    elif 'internlm' in args.model:
        model = LmdeployModel(args)
    elif 'qwen' in args.model.lower():
        model = LmdeployModel(args)
    else:
        logger.info(f"{args.model} not supported!")
        exit()
    return model


if __name__ == '__main__':
    from dataclasses import dataclass, field
    from transformers import (
        HfArgumentParser, 
        AutoTokenizer
    )
    from arguments import GlobalArguments
    
    
    @dataclass
    class ModelArguments(GlobalArguments):
        model_url: str = None
        model_path: str = None
        tp: int = 1

    parser = HfArgumentParser(ModelArguments)
    # parser.parse_args_into_dataclasses()
    args = parser.parse_args_into_dataclasses()[0]
    llm = LLama3ModelClient(args)
    
    inputs = [
        {
            "pair_id": 0,
            "system": "你是一个智能体",
            "examples": [
            {
                "user": "你好",
                "assistant": "你好，有什么我可以帮你的？"
            },
            {
                "user": "上海今天天气怎么样？",
                "assistant": "对不起，我无法获取实时天气信息。"
            },
            {
                "user": "帮我写一个五言律诗",
                "assistant": "山不在高..."
            }
            ],
            "prompt": "拜拜"
        }
    ]
    llm.run_predictions(inputs, args)