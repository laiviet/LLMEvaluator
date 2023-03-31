from flask import Flask, request, jsonify
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def make_pipeline():
    MODEL_NAME_OR_PATH = 'gpt2'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading model:', MODEL_NAME_OR_PATH)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        padding_side='left'
    )

    model.to(DEVICE)
    p = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_new_tokens=100,
        pad_token_id=model.config.eos_token_id
    )
    p.tokenizer.pad_token = model.config.eos_token_id

    return p


_pipeline = make_pipeline()


def generate(prompts):
    """

    :param prompts:
    :return:
    """
    batch_result_texts = _pipeline(prompts, batch_size=len(prompts))
    texts = [x['generated_text']
             for batch in batch_result_texts
             for x in batch]
    answers = [t[len(p):].strip() for t, p in zip(texts, prompts)]
    return answers

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    json_in = json.loads(request.data.decode(encoding='utf-8'))
    prompts = json_in['prompt']
    responses = generate(prompts)
    json_in['output'] = responses
    return jsonify(json_in)



if __name__ == "__main__":
    app.run(host='localhost', port='8888')
