import time
import openai
import pandas as pd
from tqdm import tqdm

import transformers
import torch
from pathlib import Path

class EasyCallModel:

    def __init__(self, model, tok, system=True):
        self.device = torch.device('cuda')
        self.system = system
        try:
            self.model = model.to(self.device)
        except:
            print('Skipping cast due to quantization')
            self.model = model
        self.tok = tok
    
    def __call__(self, prompt, max_tokens=256, temp=0.7, p=1):
        messages = [
            {"role": "system", "content" : 'You are a helpful assistant.'},
            {"role": "user", "content": prompt}
        ]
        if not self.system:
            messages = [messages[-1]]
        prompt = self.tok.apply_chat_template(messages, tokenize=False)
        without_label = self.tok(prompt)
        sample = self.model.generate(
                input_ids=torch.tensor(without_label['input_ids']).reshape(1, -1).to(self.device),
                attention_mask=torch.tensor(without_label['attention_mask']).reshape(1, -1).to(self.device),
                max_new_tokens=max_tokens,
                do_sample=True,
                use_cache=True,
                temperature=temp,
                top_p=p
            )
        sample = sample[0].tolist()
        decoded = self.tok.decode(sample)
        return decoded

def get_hf_model(model, load_in_4bit=True, system=True):
    token = None
    if load_in_4bit:
        tok = transformers.AutoTokenizer.from_pretrained(model, padding_side='left', token=token)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,
            load_in_4bit=True,
            token=token,
            trust_remote_code=True)

    else:
        tok = transformers.AutoTokenizer.from_pretrained(model, padding_side='left', token=token)
        model = transformers.AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, token=token)
    
    return EasyCallModel(model, tok, system=system)
    
def call_hf_model(model, prompt, temp=0.7, p=1, max_tokens=256, delay=None):
    return model(prompt, max_tokens=max_tokens, temp=temp, p=p)

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# Define a function that adds a delay to a Completion API call
def delayed_chat_call(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    if delay_in_seconds is not None:
        time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.ChatCompletion.create(**kwargs)

def call_chat_gpt(prompt, temp=0.7, p=1, delay=None, max_tokens=256):
    completion = delayed_chat_call(delay,
        model="gpt-3.5-turbo",
        temperature=temp,
        top_p=p,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message['content']

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    if delay_in_seconds is not None:
        time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.Completion.create(**kwargs)

def gpt3response(prompt, p=0.95):
    
    # Calculate the delay based on your rate limit
    # rate_limit_per_minute = 2_950
    # delay = 60.0 / rate_limit_per_minute
    delay = None
    
    completion = delayed_completion(
        model="text-davinci-002",
        delay_in_seconds=delay,
        prompt=prompt,
        max_tokens=256,
        top_p=p,
        temperature=1
    )

    return completion['choices'][0]['text']

def run_test(df, closure, cols, stop=None, p=0.95, chat=False, hf_model=None):
    data = tqdm(list(zip(df.index, *(df[c] for c in cols))))
    prompts = []
    responses = []
    for i, args in enumerate(data):
        idx = args[0]
        args = args[1:]
        if stop is not None and i > stop:
            break
        prompt = closure(*args)
        try:
            if chat:
                response = call_chat_gpt(prompt)
            elif hf_model is not None:
                response = call_hf_model(hf_model, prompt)
            else:
                response = gpt3response(prompt, p=p)
        except Exception as e:
            with open('gpt-test-errors.txt', 'a') as out:
                out.write(str(e) + '\n')
                out.write(f'index: {idx}\n')
                out.write(f'prompt: {prompt}\n')
            response = None
        prompts.append(prompt)
        responses.append(response)
    if stop is not None:
        df = df.head(stop+1)
    df['prompt'] = prompts
    df['responses'] = responses
    return df # only needed because sometimes df -> df.head

def aoa_prompt(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += f'the word that most means "{args[4]}" is'
    return prompt

def wax_prompt(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'the two words that go together best are'
    return prompt

if __name__ == '__main__':

    chat = False
    # TODO: add code for getting open source results
    hf_model_str = None
    if hf_model_str is not None:
        chat = False
        print('Resetting chat')
    hf_model = None if hf_model_str is None else get_hf_model(hf_model_str)

    # parser.add_argument('--apikey', type=str, help='API KEY')
    # parser.add_argument('--orgkey', type=str, help='ORG KEY')
    # args = parser.parse_args()
    
    # openai.api_key = # args.apikey
    # openai.organization = # args.orgkey
    aoa = pd.read_csv('tests/aoa-test.csv')
    wax = pd.read_csv('tests/wax-test.csv')
    cols = ['test0', 'test1', 'test2', 'test3']

    # test run
    # aoa = aoa.sample(frac=0.1, random_state=515)
    # omitting because done
    # aoa = run_test(aoa, aoa_prompt, cols + ['def'], p=0.95)
    # aoa.to_csv('results/aoa-raw.csv') # should we be careful about quoting here?
    # test run
    wax = wax.sample(frac=0.001, random_state=515)

    wax = run_test(wax, wax_prompt, cols, p=0.95, chat=chat, hf_model=hf_model)
    if chat and hf_model_str is None:
        wax.to_csv('results-chat/wax-raw.csv')
    elif hf_model_str is not None:
        hf_model_str = hf_model_str.replace('/', '+')
        Path(f'results-{hf_model_str}').mkdir(exist_ok=True)
        wax.to_csv(f'results-{hf_model_str}/wax-raw.csv')
    else:
        wax.to_csv('results/wax-raw.csv')