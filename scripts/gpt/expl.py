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

if __name__ == '__main__':
    wax = pd.read_csv('results/wax-features.csv')
    wax['count'] = 1

    # plot bar plot
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x='exp', y='count', hue='incorrect', 
        estimator=sum, data=wax, ax=ax)
    plt.xticks(rotation=10)
    plt.title('Count of Errors v. Expl.')
    plt.ylabel('Count')
    plt.xlabel('Explanation', fontweight="bold")
    plt.tight_layout()
    plt.savefig('results/expl-bar')

    # plot stacked percentages
    # https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot
    # from raw value to percentage
    fig, ax = plt.subplots(figsize=(1.4, 3))
    total = wax.groupby('exp')['count'].sum().reset_index()
    incorrect = wax[wax.incorrect==1].groupby('exp')['count'].sum().reset_index()
    incorrect['count'] = [i / j * 100 for i,j in zip(incorrect['count'], total['count'])]
    total['count'] = [i / j * 100 for i,j in zip(total['count'], total['count'])]

    # bar chart 1 -> top bars 
    bar1 = sns.barplot(x="exp",  y="count", data=total, color='lightslategrey')

    # bar chart 2 -> bottom bars
    bar2 = sns.barplot(x="exp", y="count", data=incorrect, color='pink')

    # add legend
    top_bar = mpatches.Patch(color='lightslategrey', label='co')
    bottom_bar = mpatches.Patch(color='pink', label='er')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xticks(rotation=30)
    plt.ylabel(None)
    plt.title('Explanation', fontweight="bold")
    plt.xlabel(None)
    # plt.title('Percent error v. explanation')
    plt.tight_layout(rect=[-0.13, -0.01, 1.13, 1.1])
    plt.savefig('results/expl-sbar')

    # chi2test
    counts0 = wax[wax['incorrect']==0]['exp']\
        .value_counts().to_dict()
    counts1 = wax[wax['incorrect']==1]['exp']\
        .value_counts().to_dict()
    f_obs = defaultdict(list)
    for k,v in counts0.items():
        f_obs[k].append(v)
    for k,v in counts1.items():
        f_obs[k].append(v)
    f_obs = np.array(list(f_obs.values()))
    print(chisquare(f_obs))
