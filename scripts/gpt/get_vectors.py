# imports
# import argparse
import time
import openai
import pandas as pd
from tqdm import tqdm

def gpt_embedding(text):
    return openai.Embedding.create(input=[text], 
        model='text-embedding-ada-002')['data'][0]['embedding']

def safe_gpt_embedding(idx, text):
    try:
        embd = gpt_embedding(humdial)
    except Exception as e:
        with open('results/gpt-vector-errors.txt', 'a') as out:
            out.write(str(e) + '\n')
            out.write(f'index: {idx}\n')
            out.write(f'text: {text}\n')
        embd = None
    return embd

def make_human_dial(row):
    dial = row.prompt
    dial += f' "{row.cue}" and "{row.association}".'
    dial += f' {row.explanation}'
    return dial

def make_gpt_dial(row):
    dial = row.prompt + row.responses
    return dial

if __name__ == '__main__':
    # parser.add_argument('--apikey', type=str, help='API KEY')
    # parser.add_argument('--orgkey', type=str, help='ORG KEY')
    # args = parser.parse_args()
    # openai.api_key = # args.apikey
    # openai.organization = # args.orgkey
    wax = pd.read_csv('results/wax-features.csv')
    wax = wax[wax['exp']]
    wax['human_dial'] = wax.apply(make_human_dial, axis=1)
    wax['gpt_dial'] = wax.apply(make_gpt_dial, axis=1)
    human_embds = []
    gpt_embds = []
    data = tqdm(list(zip(wax.index, wax['human_dial'], wax['gpt_dial'])))
    # i = 0
    for idx, humdial, gptdial in data:
        # i +=1
        # if i > 100:
        #     break
        human_embds.append(safe_gpt_embedding(idx, humdial))
        gpt_embds.append(safe_gpt_embedding(idx, gptdial))
    # originally for testing, but don't remove just in case
    wax = wax.head(len(human_embds)).copy()
    wax['hum_embd'] = human_embds
    wax['gpt_embd'] = gpt_embds
    wax.to_csv('results/wax-vectors.csv')
