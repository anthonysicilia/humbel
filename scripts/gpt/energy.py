import pandas as pd
from discrete_energy.statistic import DiscreteEnergyStatistic
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import sqrt
from scripts.gpt.errors import wax_gt

random.seed(2022515)
plt.rcParams.update({'font.size': 22})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def str_to_vec(s):
    try:
        elems = [c for c in s.strip('][').split(', ')]
        elems = [float(c) for c in elems]
        return elems
    except:
        return None

if __name__ == '__main__':

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2), sharey=True)

    eq = lambda x,y: x == y
    leq = lambda x,y: x <= y
    wax = pd.read_csv('results/wax-vectors.csv')
    wax['gt'] = wax.apply(wax_gt, axis=1)
    wax['hum_embd'] = wax['hum_embd'].map(str_to_vec)
    wax['gpt_embd'] = wax['gpt_embd'].map(str_to_vec)
    wax = wax[~wax['hum_embd'].isna()]
    wax = wax[~wax['gpt_embd'].isna()]
    train = np.array([random.random() < 0.5 for _ in wax.index])
    test = wax[~train].copy()
    train = wax[train].copy()
    del wax

    for i, (comparison, split_col) in enumerate([(eq, 'pair_pos'), (leq, 'max_aoa_pair')]):
        jitter = 0.005 if split_col == 'pair_pos' else 0.0005
        counts = train[split_col].value_counts()
        col_vals = set(counts[counts > 100].index)
        energy = []
        testdiv = []
        for val in tqdm(col_vals):
            idx = comparison(test[split_col], val)
            n = idx.sum()
            correct = (test[idx]['gt'] == test[idx]['intent']).astype(int).values
            td = correct.sum() / n
            train_idx = comparison(train[split_col], val)
            train_s1 = train[train_idx]['hum_embd'].tolist()
            train_s2 = train[train_idx]['gpt_embd'].tolist()
            stat = DiscreteEnergyStatistic(n, n, learn_clusters=True,
                nclusters=min(int(.05 * n), 50), seed=515, 
                train_sample_1=torch.tensor(train_s1),
                train_sample_2=torch.tensor(train_s2))
            # test to perfect figure
            # stat = lambda *args: random.random()
            test_s1 = train[train_idx]['hum_embd'].tolist()
            test_s2 = train[train_idx]['gpt_embd'].tolist()
            e = stat(torch.tensor(test_s1), torch.tensor(test_s2))
            testdiv.append(td)#  / num_samples)
            energy.append(sqrt(e))# / num_samples)
        if i == 1: # cheat for color change
            ax.flat[i].plot([], [])
            ax.flat[i].scatter([], [])
        ax.flat[i].scatter(energy, testdiv)
        x = np.array(energy)
        y = np.array(testdiv)
        a, b = np.polyfit(x, y, 1)
        ax.flat[i].plot(x, a*x+b, lw=3, ls=':')
        # for j, txt in enumerate(col_vals):
            # ax.flat[i].annotate(txt, (energy[j], testdiv[j]),
            #     fontsize=10)
        ax.flat[i].set_xlim((x.min() - jitter, x.max() + jitter))
        ax.flat[i].set_xlabel('energy')
        if  i == 0:
            ax.flat[i].set_ylabel('test div.')
        s = 'POS' if split_col == 'pair_pos' else 'AoA'
        ax.flat[i].set_title(f'split = {s}', fontweight="bold")

    plt.tight_layout(rect=[-0.07, -0.13, 1.05, 1.13])
    plt.savefig(f'results/energy')
