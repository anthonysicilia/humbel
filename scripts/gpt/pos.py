from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import seaborn as sns; sns.set(style='whitegrid')

plt.rcParams.update({'font.size': 22})

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def filter_fn(wax):
    counts = wax['pair_pos'].value_counts().to_dict()
    keep = wax['pair_pos'].map(counts) >= 100
    return wax[keep]

if __name__ == '__main__':
    wax = pd.read_csv('results/wax-features.csv')

    # filter infreq. occurences
    counts = wax['pair_pos'].value_counts().to_dict()
    keep = wax['pair_pos'].map(counts) >= 100
    wax = wax[keep]
    wax['count'] = 1

    # plot bar plot
    fig, ax = plt.subplots(figsize=(15,4))
    sns.barplot(x='pair_pos', y='count', hue='incorrect', 
        estimator=sum, data=wax, ax=ax)
    plt.xticks(rotation=10)
    plt.title('Count of pair POS')
    plt.ylabel('Count')
    plt.xlabel('Pair POS')
    plt.tight_layout()
    plt.savefig('results/pos-bar')

    # plot stacked percentages
    # https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot
    # from raw value to percentage
    fig, ax = plt.subplots(figsize=(6, 3))
    total = wax.groupby('pair_pos')['count'].sum().reset_index()
    incorrect = wax[wax.incorrect==1].groupby('pair_pos')['count'].sum().reset_index()
    incorrect['count'] = [i / j * 100 for i,j in zip(incorrect['count'], total['count'])]
    total['count'] = [i / j * 100 for i,j in zip(total['count'], total['count'])]

    # bar chart 1 -> top bars 
    bar1 = sns.barplot(x="pair_pos",  y="count", data=total, color='royalblue')

    # bar chart 2 -> bottom bars
    bar2 = sns.barplot(x="pair_pos", y="count", data=incorrect, color='darkorange')

    # add legend
    top_bar = mpatches.Patch(color='royalblue', label='correct')
    bottom_bar = mpatches.Patch(color='darkorange', label='error')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xticks(rotation=30)
    plt.ylabel('Percent')
    plt.title('Pair POS', fontweight="bold")
    plt.xlabel(None)
    # plt.title('Percent error for pair POS')
    plt.tight_layout(rect=[-0.03, -0.1, 1.03, 1.1])
    plt.savefig('results/pos-sbar')

    # chi2test
    counts0 = wax[wax['incorrect']==0]['pair_pos']\
        .value_counts().to_dict()
    counts1 = wax[wax['incorrect']==1]['pair_pos']\
        .value_counts().to_dict()
    f_obs = defaultdict(list)
    for k,v in counts0.items():
        f_obs[k].append(v)
    for k,v in counts1.items():
        f_obs[k].append(v)
    f_obs = np.array(list(f_obs.values()))
    print(chisquare(f_obs))
