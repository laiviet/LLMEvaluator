"""
Author: Viet Lai
"""

from transformers import *
from transformers.pipelines import TextGenerationPipeline, Text2TextGenerationPipeline
import torch
import requests
import json


class ModelWrapper():

    def __init__(self):
        super(ModelWrapper, self).__init__()

    def generate(self, prompts):
        """

        :param prompts: list of strings
        :return:
        """
        raise NotImplemented


class CausalLMModelWrapper(ModelWrapper):

    def __init__(self,
                 model_name_or_path: str,
                 model=None,
                 tokenizer=None,
                 device=torch.device('cuda:0')
                 ):
        super(CausalLMModelWrapper, self).__init__()
        if not model:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side='left'
            )

        model.to(device)
        self.pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=10,
            pad_token_id=model.config.eos_token_id
        )
        self.pipeline.tokenizer.pad_token = model.config.eos_token_id

    def generate(self, prompts):
        batch_result_texts = self.pipeline(prompts, batch_size=len(prompts))

        texts = [x['generated_text']
                 for batch in batch_result_texts
                 for x in batch]

        answers = [t[len(p):].strip() for t, p in zip(texts, prompts)]

        return answers


class ProxyClient(ModelWrapper):

    def __init__(self, endpoint='http://localhost:8888'):
        super(ProxyClient, self).__init__()
        self.endpoint = endpoint

    def generate(self, prompts):
        data = {'prompt': prompts}  # Replace with the JSON data you want to send
        response = requests.post(self.endpoint, json=data)
        if response.status_code == 200:  # Check if the response was successful
            response_json = response.json()  # Parse the response JSON
            return response_json

        return None


def test_proxy_client():
    c = ProxyClient(endpoint='http://localhost:8888')
    result = c.generate(['The world is in danger', 'What is the meaning of NSA?'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    test_proxy_client()
