import pandas as pd
from scripts.gpt.interpreter import answer_format
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
import matplotlib.pyplot as plt
from scipy.stats import binom_test

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def wax_gt(row):
    pair = [row.cue.lower(), row.association.lower()]
    return answer_format(pair)

def aoa_gt(row):
    return answer_format([row.word.lower()])

if __name__ == '__main__':

    # chat = True
    chat = False

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,2))

    leq = lambda x,y: x <= y
    eq = lambda x,y: x == y

    # math
    # 50 percent know

    # w/o hoeffding
    # Pr(know and agree with annot) 
    # = Pr(know) * Pr(agree)
    # = 0.5 * (1 - 0.15)
    # = .425

    # 51 - 49 / 6 = 42.8
    # so answer is 51
    # same but w/ Pr(know) = 0.7 -> 66

    # w/hoeffding
    # Pr(know and agree with annot) 
    # = Pr(know) * Pr(agree)
    # = 0.5 * (1 - 0.15 - 0.09) 
    # = .38

    # 47 - 53 / 6 = 38.1
    # so answer is 47
    # same but w/ Pr(know) = 0.7 -> 60
    # end math

    # wax_gamma = 0.51
    wax_gamma = 0.47
    # wax_gamma = 0.66
    # wax_gamma = 0.6
    aoa_gamma = 0.58

    for s, comp in [('$\leq$AoA', leq), ('= AoA', eq)]:

        wax_pvals = []
        wax_ages = []

        # print('wax data ========')
        if chat:
            wax = pd.read_csv('results-chat/wax-interpreted.csv')
        else:
            wax = pd.read_csv('results/wax-interpreted.csv')
        
        wax['gt'] = wax.apply(wax_gt, axis=1)
        for age in set(wax['max_aoa_pair']):
            idx = comp(wax['max_aoa_pair'], age)
            correct = (wax[idx]['gt'] == wax[idx]['intent']).sum()
            p = binom_test(correct, idx.sum(), wax_gamma, alternative='less')
            wax_pvals.append(p)
            wax_ages.append(age)
            print('wax', s, p, age)
        # incorrect = (wax['gt'] != wax['intent']).sum() / len(wax)
        ls = ':' if '=' in s else '-'
        ax.plot(wax_ages, wax_pvals, lw=3, label=s, ls=ls)
    ax.axhline(0.05, c='r', ls='--')
    ax.set_title('GPT-3.5 % Mean Test (WC large)',  fontweight="bold")
    ax.set_ylabel('p val')
    ax.set_xlabel('AoA')
    ax.legend()
    # ax.flat[0].legend()
        # ax.flat[0].set_ylabel('Accuracy')
    
    # end early, none significant for aoa
    plt.tight_layout(rect=[-0.03, -0.1, 1.04, 1.1])
    if chat:
        plt.savefig('results-chat/humsamp-pvals')
    else:
        plt.savefig('results/humsamp-pvals-new')
    # plt.savefig('results/humsamp-pvals-noub')
    exit()
    for s, comp in [('$\leq$AoA', leq), ('= AoA', eq)]:

        aoa_pvals = []
        aoa_ages = []

        # print('aoa data ========')
        aoa = pd.read_csv('results/aoa-interpreted.csv')
        aoa['gt'] = aoa.apply(aoa_gt, axis=1)
        for age in set(aoa['aoa']):
            idx = comp(aoa['aoa'], age)
            correct = (aoa[idx]['gt'] == aoa[idx]['intent']).sum()
            p = binom_test(correct, idx.sum(), aoa_gamma, alternative='less')
            aoa_pvals.append(p)
            aoa_ages.append(age)
            print('aoa', s, p, age)
        # incorrect = (aoa['gt'] != aoa['intent']).sum() / len(aoa)
        # ax.flat[1].plot([],[]) # cheat to change color
        ls = ':' if '=' in s else '-'
        ax.flat[1].plot(aoa_ages, aoa_pvals, lw=3, label=s, ls=ls)
    ax.flat[1].axhline(0.01, c='r', ls='--')
    ax.flat[1].set_title('GPT-3 % Accuracy (Def)', fontweight="bold")
    ax.flat[1].set_xlabel('AoA')
        # ax.flat[1].set_ylabel('Accuracy')
        # ax.flat[1].set_ylim((0.3, 1.1))
    ax.flat[1].legend()
    plt.tight_layout(rect=[-0.03, -0.1, 1.05, 1.1])
    plt.savefig('results/humsamp-pvals')