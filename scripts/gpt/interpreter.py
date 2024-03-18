from string import punctuation
import pandas as pd
import re

SPECIAL_DELIM = '__$$$$__'

def answer_format(answers):
    if len(answers) > 0:
        return SPECIAL_DELIM.join(sorted(answers))
    else:
        return None

def reverse_format(answer):
    try:
        x, y = answer.split(SPECIAL_DELIM)
    except ValueError:
        return 'UNK UNK'
    return f'{x} {y}'

def interpreter(row, n):
    tests = re.findall(r'"(.*?)"', row.prompt)
    tests = set([t.lower() for t in tests])
    enum_response = list(reversed(row.responses.split()))
    answers = []
    while len(answers) < n and enum_response:
        next_word = enum_response.pop()\
            .strip(punctuation).lower()
        if next_word in tests:
            answers.append(next_word)
    return answer_format(answers)

if __name__ == '__main__':

    chat = True
    # hf_model_str = 'mistralai+Mistral-7B-Instruct-v0.2'
    # hf_model_str = 'meta-llama+Llama-2-7b-chat-hf'
    # hf_model_str = 'HuggingFaceH4+zephyr-7b-beta'
    # hf_model_str = 'chat-icl'
    hf_model_str = 'chat-icl-10'

    if chat and hf_model_str is None:
        wax = pd.read_csv('results-chat/wax-raw.csv')
        wax['responses'] = wax['responses'].astype(str)
    elif hf_model_str is not None:
        wax = pd.read_csv(f'results-{hf_model_str}/wax-raw.csv')
        wax['responses'] = wax['responses'].astype(str)
        wax['responses'] = wax['responses'].apply(lambda s: s.split('[/INST]')[-1])
        wax['responses'] = wax['responses'].apply(lambda s: s.split('<|assistant|>')[-1])
    else:
        wax = pd.read_csv('results/wax-raw.csv')
    wax_interpreter = lambda x: interpreter(x, 2)
    aoa_interpreter = lambda x: interpreter(x, 1)
    wax['intent'] = wax.apply(wax_interpreter, axis=1)
    b4 = len(wax)
    wax = wax[~wax[f'intent'].isna()]
    print('Dropped from wax:', b4 -len(wax))
    if chat and hf_model_str is None:
        wax.to_csv('results-chat/wax-interpreted.csv')
    elif hf_model_str is not None:
        wax.to_csv(f'results-{hf_model_str}/wax-interpreted.csv')
    else:
        wax.to_csv('results/wax-interpreted.csv')
        aoa = pd.read_csv('results/aoa-raw.csv')
        b4 = len(aoa)
        aoa = aoa[~aoa[f'responses'].isna()]
        print('Dropped from aoa:', b4 -len(aoa))
        aoa['intent'] = aoa.apply(aoa_interpreter, axis=1)
        aoa.to_csv('results/aoa-interpreted.csv')
        print(wax)
        print(aoa)

