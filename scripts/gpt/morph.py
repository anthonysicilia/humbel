from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import seaborn as sns; sns.set(style='whitegrid')
import spacy

plt.rcParams.update({'font.size': 22})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def morph_cats(x):
    if x <= 2:
        return 'low'
    elif x > 2 and x <= 4:
        return 'med'
    else:
        return 'high'

if __name__ == '__main__':
    wax = pd.read_csv('results/wax-features.csv')
    wax['num_morphs'] = wax['num_morphs'].map(morph_cats)
    wax['count'] = 1

    # plot bar plot
    fig, ax = plt.subplots(figsize=(2.2,3))
    sns.barplot(x='num_morphs', y='count', hue='incorrect', 
        estimator=sum, data=wax, ax=ax)
    plt.yscale('log')
    plt.xticks(rotation=25)
    plt.title('Count of Num. Morphs.')
    plt.tight_layout()
    plt.savefig('results/morph-bar')

    # plot stacked percentages
    # https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot
    # from raw value to percentage
    fig, ax = plt.subplots(figsize=(1.8, 3))
    total = wax.groupby('num_morphs')['count'].sum().reset_index()
    incorrect = wax[wax.incorrect==1].groupby('num_morphs')['count'].sum().reset_index()
    incorrect['count'] = [i / j * 100 for i,j in zip(incorrect['count'], total['count'])]
    total['count'] = [i / j * 100 for i,j in zip(total['count'], total['count'])]

    # bar chart 1 -> top bars 
    bar1 = sns.barplot(x="num_morphs",  y="count", data=total, color='peru')

    # bar chart 2 -> bottom bars
    bar2 = sns.barplot(x="num_morphs", y="count", data=incorrect, color='rebeccapurple')

    # add legend
    top_bar = mpatches.Patch(color='peru', label='correct')
    bottom_bar = mpatches.Patch(color='rebeccapurple', label='error')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xticks(rotation=30)
    plt.ylabel(None)
    plt.title('Num. Morphs', fontweight="bold")
    plt.xlabel(None)
    # plt.title('Percent error for Num. Morphs.')
    plt.tight_layout(rect=[-0.1, 0.0, 1.1, 1.1])
    plt.savefig('results/morph-sbar')

    # chi2test
    counts0 = wax[wax['incorrect']==0]['num_morphs']\
        .value_counts().to_dict()
    counts1 = wax[wax['incorrect']==1]['num_morphs']\
        .value_counts().to_dict()
    f_obs = defaultdict(list)
    for k,v in counts0.items():
        f_obs[k].append(v)
    for k,v in counts1.items():
        f_obs[k].append(v)
    f_obs = np.array(list(f_obs.values()))
    print(chisquare(f_obs))