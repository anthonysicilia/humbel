# imports
# import argparse
import time
import openai
import pandas as pd
from tqdm import tqdm

from scipy.stats import f_oneway, kruskal, chisquare
import statsmodels.formula.api as smf

from .interpreter import interpreter
from .errors import wax_gt

# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    if delay_in_seconds is not None:
        time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.Completion.create(**kwargs)

def gpt3response(prompt, p=0.95, temp=1):
    
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
        temperature=temp
    )

    return completion['choices'][0]['text']

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# Define a function that adds a delay to a Completion API call
def delayed_chat_call(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    if delay_in_seconds is not None:
        time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.ChatCompletion.create(**kwargs)

def call_chat_gpt(prompt, temp=1, p=0.95, delay=None, max_tokens=256):
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

def wax_prompt_0(*args):
    prompt = 'Listen carefully to the words I say, then tell me the two words that go together best: '
    prompt += f'"{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}".'
    return prompt

def wax_prompt_1(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'what are the two words that go together best?'
    return prompt

def wax_prompt_2(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'What are the two words that go together best?'
    return prompt

def wax_prompt_3(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'what two words that go together best?'
    return prompt

def wax_prompt_4(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'What two words that go together best?'
    return prompt

def wax_prompt_5(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'pick the two words that go together best.'
    return prompt

def wax_prompt_6(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'Pick the two words that go together best?'
    return prompt

def wax_prompt_7(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'pick the two words that go together best and give an explanation.'
    return prompt

def wax_prompt_8(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'Pick the two words that go together best and provide an explanation.'
    return prompt

def wax_prompt_9(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'what are the two words that go together best? Why?'
    return prompt

def wax_prompt_10(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'What are the two words that go together best? Why?'
    return prompt

# instruct prompts
def waxi_prompt_0(*args):
    prompt = 'Listen carefully to the words I say, then tell me the two words that go together best: '
    prompt += f'"{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}".'
    return prompt

def waxi_prompt_1(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'the two words that go together best are'
    return prompt

def waxi_prompt_2(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'Tell me the two words that go together best:'
    return prompt

def waxi_prompt_3(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'tell me the two words that go together best:'
    return prompt

def waxi_prompt_4(*args):
    prompt = f'Instructor: Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'What two words that go together best?\nStudent:'
    return prompt

def waxi_prompt_5(*args):
    prompt = f'Instructor: Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'pick the two words that go together best.\nStudent:'
    return prompt

def waxi_prompt_6(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'Pick the two words that go together best.'
    return prompt

def waxi_prompt_7(*args):
    prompt = f'Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'pick the two words that go together best and give an explanation.'
    return prompt

def waxi_prompt_8(*args):
    prompt = f'Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'Pick the two words that go together best and provide an explanation.'
    return prompt

def waxi_prompt_9(*args):
    prompt = f'Instructor: Among the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}", '
    prompt += 'what are the two words that go together best? Why?\nStudent:'
    return prompt

def waxi_prompt_10(*args):
    prompt = f'Instructor:Consider the words "{args[0]}", "{args[1]}", "{args[2]}", and "{args[3]}". '
    prompt += 'What are the two words that go together best? Why?\nStudent:'
    return prompt

wax_prompts = [
    wax_prompt_0,
    wax_prompt_1,
    wax_prompt_2,
    wax_prompt_3,
    wax_prompt_4,
    wax_prompt_5,
    wax_prompt_6,
    wax_prompt_7,
    wax_prompt_8,
    wax_prompt_9,
    wax_prompt_10
]

wax_instruct_prompts =[
    waxi_prompt_0,
    waxi_prompt_1,
    waxi_prompt_2,
    waxi_prompt_3,
    waxi_prompt_4,
    waxi_prompt_5,
    waxi_prompt_6,
    waxi_prompt_7,
    waxi_prompt_8,
    waxi_prompt_9,
    waxi_prompt_10
]

parameters = [
    {
        'p' : 1,
        'temp' : 1,

    },
    {
        'p' : 0.95,
        'temp' : 1,

    },
    {
        'p' : 0.9,
        'temp' : 1,

    },
    {
        'p' : 0.8,
        'temp' : 1,

    },
    {
        'p' : 1,
        'temp' : 0.7,

    },
    {
        'p' : 1,
        'temp' : 0.5,

    },
    {
        'p' : 1,
        'temp' : 0,

    }
]

def run_test(df, closure, cols, stop=None, p=0.95, temp=1, instruct=False):
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
            if instruct:
                response = gpt3response(prompt, p=p, temp=temp)
            else:
                response = call_chat_gpt(prompt, p=p, temp=temp)
        except Exception as e:
            with open('gpt-sensi-errors.txt', 'a') as out:
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

def get():

    instruct = True

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--apikey', type=str, help='API KEY')
    # parser.add_argument('--orgkey', type=str, help='ORG KEY')
    # args = parser.parse_args()
    # openai.api_key = # args.apikey
    # openai.organization = # args.orgkey
    wax = pd.read_csv('tests/wax-test.csv')
    seed = 0
    wax = wax.sample(n=100, random_state=seed)
    res = None
    cols = ['test0', 'test1', 'test2', 'test3']
    settings = [(prompt, params) for prompt in wax_prompts
        for params in parameters]
    for i, (prompt, params) in enumerate(settings):
        print(f'Running {i}/{len(settings)}...')
        w = run_test(wax, prompt, cols, instruct=instruct, **params)
        w['setting'] = i
        if res is None:
            res = w
        else:
            res = pd.concat([res, w])
        if instruct:
            res.to_csv('results/wax-sensi.csv')
        else:
            res.to_csv('results-chat/wax-sensi.csv')

def evaluate():
    wax = pd.read_csv('results/wax-sensi.csv')
    wax['responses'] = wax['responses'].astype(str)
    wax_interpreter = lambda x: interpreter(x, 2)
    wax['intent'] = wax.apply(wax_interpreter, axis=1)
    b4 = len(wax)
    wax = wax[~wax[f'intent'].isna()]
    print('Dropped from wax sensi:', b4 - len(wax))
    wax['gt'] = wax.apply(wax_gt, axis=1)
    wax['correct'] = wax['gt'] == wax['intent']
    sensi_agg = wax.groupby('setting')['correct'].aggregate('mean')
    print('n', len(wax))
    print('m', len(sensi_agg))
    print('Mean:', sensi_agg.mean())
    print('Standard Deviation:', sensi_agg.std())
    samples = []
    for i in set(wax['setting']):
        # samples.append(wax[wax['setting'] == i]['correct'].values)
        samples.append(wax[wax['setting'] == i]['correct'].sum())
    # print('One Way ANOVA:', f_oneway(*samples))
    # print('Kruskal:', kruskal(*samples))
    print('Chi square:', chisquare(samples))
    # if sig check logits, no interaction model
    # wax['correct'] = wax['correct'].astype(int)
    # wax['setting'] = wax['setting'].astype(str)
    # formula = 'correct ~ setting'
    # lreg = smf.logit(formula, data=wax.dropna()).fit()
    # print('Logit Model')
    # print(lreg.summary())




if __name__ == '__main__':
    # get()
    evaluate()
