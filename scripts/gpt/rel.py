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
    counts = wax['relation'].value_counts().to_dict()
    keep = wax['relation'].map(counts) >= 25
    return wax[keep]

if __name__ == '__main__':
    wax = pd.read_csv('results/wax-features.csv')

    # filter infreq. occurences
    wax = filter_fn(wax)
    wax['count'] = 1
    shorten = lambda s: 'Phrase 'if s == 'Common-Phrase' else s
    wax['relation'] = wax['relation'].map(shorten)

    # plot bar plot
    fig, ax = plt.subplots(figsize=(15,4))
    wax_no_unk = wax[wax['relation'] != 'UNK']
    sns.barplot(x='relation', y='count', hue='incorrect', 
        estimator=sum, data=wax_no_unk, ax=ax)
    plt.xticks(rotation=10)
    plt.title('Count of WAX Relations (UNK removed)')
    plt.tight_layout()
    plt.savefig('results/rel-bar')

    # plot stacked percentages
    # https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot
    # from raw value to percentage
    fig, ax = plt.subplots(figsize=(3.4, 3))
    total = wax.groupby('relation')['count'].sum().reset_index()
    incorrect = wax[wax.incorrect==1].groupby('relation')['count'].sum().reset_index()
    incorrect['count'] = [i / j * 100 for i,j in zip(incorrect['count'], total['count'])]
    total['count'] = [i / j * 100 for i,j in zip(total['count'], total['count'])]

    # bar chart 1 -> top bars 
    bar1 = sns.barplot(x="relation",  y="count", data=total, color='mediumseagreen')

    # bar chart 2 -> bottom bars
    bar2 = sns.barplot(x="relation", y="count", data=incorrect, color='indianred')

    # add legend
    top_bar = mpatches.Patch(color='mediumseagreen', label='correct')
    bottom_bar = mpatches.Patch(color='indianred', label='error')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xticks(rotation=30)
    plt.ylabel(None)
    plt.title('Relation', fontweight="bold")
    plt.xlabel(None)
    # plt.title('Percent error for Relations')
    plt.tight_layout(rect=[-0.07, -0.093, 1.07, 1.1])
    plt.savefig('results/rel-sbar')

    # chi2test on wax
    counts0 = wax[wax['incorrect']==0]['relation']\
        .value_counts().to_dict()
    counts1 = wax[wax['incorrect']==1]['relation']\
        .value_counts().to_dict()
    f_obs = defaultdict(list)
    for k,v in counts0.items():
        f_obs[k].append(v)
    for k,v in counts1.items():
        f_obs[k].append(v)
    f_obs = np.array(list(f_obs.values()))
    print('wax full:', chisquare(f_obs))

    # chi2test on wax_no_unk
    wax = wax_no_unk.copy()
    counts0 = wax[wax['incorrect']==0]['relation']\
        .value_counts().to_dict()
    counts1 = wax[wax['incorrect']==1]['relation']\
        .value_counts().to_dict()
    f_obs = defaultdict(list)
    for k,v in counts0.items():
        f_obs[k].append(v)
    for k,v in counts1.items():
        f_obs[k].append(v)
    f_obs = np.array(list(f_obs.values()))
    print('wax no unk:', chisquare(f_obs))