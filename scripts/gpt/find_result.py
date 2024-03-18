import pandas as pd

if __name__ == '__main__':
    wax = pd.read_csv('results/wax-interpreted.csv')
    assoc = None
    while assoc != '$$$$':
        cue = input('Cue:')
        assoc = input('Assoc:')
        show = wax[wax['cue']==cue]
        show = show[show['association']==assoc]
        show = zip(show['prompt'], show['responses'], show['intent'])
        for prompt, response, intent in show:
            print('Prompt:', prompt)
            print('Response:', response)
            print('Intent:', intent)