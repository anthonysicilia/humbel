import pandas as pd
from scripts.gpt.interpreter import answer_format
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def wax_gt(row):
    pair = [row.cue.lower(), row.association.lower()]
    return answer_format(pair)

def aoa_gt(row):
    return answer_format([row.word.lower()])

if __name__ == '__main__':

    chat = True

    if chat:
        ncols = 1
    else:
        ncols = 2

    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(5,2))

    leq = lambda x,y: x <= y
    eq = lambda x,y: x == y

    for s, comp in [('$\leq$AoA', leq), ('= AoA', eq)]:

        wax_errors = []
        wax_ages = []

        print('wax data ========')
        if chat:
            wax = pd.read_csv('results-chat/wax-interpreted.csv')
        else:
            wax = pd.read_csv('results/wax-interpreted.csv')
        wax['gt'] = wax.apply(wax_gt, axis=1)
        for age in set(wax['max_aoa_pair']):
            idx = comp(wax['max_aoa_pair'], age)
            # idx = wax['max_aoa_pair'] == age
            correct = (wax[idx]['gt'] == wax[idx]['intent']).sum() / idx.sum()
            print(f'\% correct wax at {age}: {correct} (n={idx.sum()})')
            wax_errors.append(100 * correct)
            wax_ages.append(age)
        correct = (wax['gt'] == wax['intent']).sum() / len(wax)
        print(f'\% correct wax overall: {correct}')
        ls = ':' if '=' in s else '-'
        if chat:
            ax.plot(wax_ages, wax_errors, lw=3, label=s, ls=ls)
        else:
            ax.flat[0].plot(wax_ages, wax_errors, lw=3, label=s, ls=ls)

    if chat:
        ax.set_title('GPT-3.5 % Acc. (WC large)',  fontweight="bold")
        ax.set_xlabel('AoA')
        ax.legend()
        ax.set_ylabel('Accuracy')
        plt.savefig('results-chat/accuracy')
        plt.tight_layout()
        exit()
    
    ax.flat[0].set_title('GPT-3.5 % Acc. (WC large)',  fontweight="bold")
    ax.flat[0].set_xlabel('AoA')


    for s, comp in [('$\leq$AoA', leq), ('= AoA', eq)]:

        aoa_errors = []
        aoa_ages = []

        print('aoa data ========')
        aoa = pd.read_csv('results/aoa-interpreted.csv')
        aoa['gt'] = aoa.apply(aoa_gt, axis=1)
        for age in set(aoa['aoa']):
            idx = comp(aoa['aoa'], age)
            # idx = aoa['aoa'] == age
            correct = (aoa[idx]['gt'] == aoa[idx]['intent']).sum() / idx.sum()
            aoa_errors.append(100 * correct)
            aoa_ages.append(age)
            print(f'\% correct aoa at {age}: {correct}')
        correct = (aoa['gt'] == aoa['intent']).sum() / len(aoa)
        print(f'\% correct aoa overall: {correct}')

        # ax.flat[1].plot([],[]) # cheat to change color
        ls = ':' if '=' in s else '-'
        ax.flat[1].plot(aoa_ages, aoa_errors, lw=3, label=s, ls=ls)
    ax.flat[1].set_title('GPT-3.5 % Acc. (Def)', fontweight="bold")
    ax.flat[1].set_xlabel('AoA')
        # ax.flat[1].set_ylabel('Accuracy')
        # ax.flat[1].set_ylim((0.3, 1.1))
    ax.flat[1].legend()
    plt.tight_layout(rect=[-0.03, -0.1, 1.05, 1.1])
    plt.savefig('results/accuracy')