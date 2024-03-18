# PROMPTS
from collections import defaultdict
from tqdm import tqdm
import openai
import argparse
import pickle
import time

def prompt_to_qa(prompt):
    return f'Instructor: {prompt}\n\nStudent:'

def make_prompts(data, slp=False, qa_format=False):
    prompts = defaultdict(list)

    for sub in data['WC'].keys():
        for x in data['WC'][sub]:
            if slp:
                prompt = 'Listen carefully to the words I say, then tell me the two words that go together best: '
                prompt += ', '.join([f'"{xi}"' for xi in x])
                prompt += '.'
            else:
                prompt = 'Among the words '
                prompt += ', '.join([f'"{xi}"' for xi in x[:-1]])
                prompt += f', and "{x[-1]}", '
                prompt += 'the two words that go together best are'
            prompts['WC'].append(prompt)

    for x in data['FS'][0]:
        if slp:
            prompt = f'Tell me a sentence using the word "{x}".'
        else:
            prompt = f'Consider the following example using the word "{x}" in a sentence'
        prompts['FS'].append(prompt)
    
    for x in data['FS'][1]:
        if slp:
            prompt = f'Tell me a sentence using the words "{x[0]}" and "{x[1]}".'
        else:
            prompt = f'Consider the following example using both of the words "{x[0]}" and "{x[1]}" in a sentence'
        prompts['FS'].append(prompt)
    
    for sub in data['RS']:
        for x in data['RS'][sub]:
            if slp:
                prompt = f'Listen to the sentence and say exactly what I say. '
                prompt += f'"{x}"'
            else:
                prompt = f'Repeat the sentence "{x}" one time'
            prompts['RS'].append(prompt)
    
    for sub in data['USP']:
        x = data['USP'][sub]
        for question in x['questions']:
            if slp:
                prompt = 'Listen carefully to the following story. Afterward I will ask you questions about what I read.\n'
                prompt += f'{x["story"]}\n'
                prompt += f'{question}'
            else:
                prompt = f'Story: {x["story"]}\n'
                prompt += f'Question: {question}'
            prompts['USP'].append(prompt)
    
    if qa_format:
        for k in prompts.keys():
            for i, p in enumerate(prompts[k]):
                prompts[k][i] = prompt_to_qa(p)

    return prompts

prompt_data = {
    'WC' : {
        0 : [
            ('dog', 'cat', 'puppy'),
            # NOTE: real test questions need to be purchased from Pearson per publishing agreement
            # NOTE: add more to these lists after purchase

        ],
        1 : [
            ('dish', 'spoon', 'plat', 'kite'),
        ]
    },
    'FS': {
        0 : [
            'he',
        ],
        1 : [
            ('although', 'still'),
        ]
    },
    'RS' : {
        0 : [
            'The kids are playing.',
        ],
    },
    'USP' : {
        0 : {
            'story' : "Story goes here...",

            'questions' : [
                'Why did ... ?',
                "How  ... ?",
                "What is  ... ",
                'What ... ',
            ]
        # NOTE: stories and questions need to be purchased from pearson

        },
        1 : {
            'story' : "Story goes here...",

            'questions' : [
                'What did ... ?',
                'When did ... ?',
                'What two  ...?'
                'When did  ...',
                "What do  ... "
            ]
        },
        # NOTE: more story QA pairs go here
    }   
}

def gpt3response(prompt):
    completion = openai.Completion.create(
        model="text-davinci-002",
        # model="text-davinci-003",
        prompt=prompt,
        max_tokens=256,
        # top_p=1,
        # temperature=0)
        # default below
        top_p=0.95,
        temperature=1)
    return completion['choices'][0]['text']

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--apikey', type=str, help='API KEY')
    # args = parser.parse_args()
    # openai.api_key = # args.apikey

    ml_prompts = make_prompts(prompt_data)
    slp_prompts = make_prompts(prompt_data, slp=True)
    slp_qa_prompts = make_prompts(prompt_data, slp=True, qa_format=True)
    prompts = [('ML', k, p) for k in ml_prompts for p in ml_prompts[k]]
    prompts += [('SLP', k, p) for k in slp_prompts for p in slp_prompts[k]]
    prompts += [('SLP QA', k, p) for k in slp_qa_prompts for p in slp_qa_prompts[k]]
    data = []
    # test = set()
    # for k in slp_prompts.keys():
    #     print(slp_prompts[k][0])
    # exit('Did you mean to run this?')
    for i, (meta1, meta2, prompt) in enumerate(tqdm(prompts)):
        # if meta2 not in test:
        # if i > 0 and i % 50 == 0:
        #     print(i, 'Sleeping...')
        #     time.sleep(70) # strategy to get around rate limit
        #     print('Continuing...')
        # if meta1 != 'ML' and meta2 != 'WC':
        #     continue
        response = gpt3response(prompt)
        data.append((meta1, meta2, prompt, response))
        # test.add(meta2)
    pickle.dump(data, open('default-slp-qa.pkl', 'wb'))



