import pandas as pd
import numpy as np
import random
from scipy.stats import binom_test, norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')

from math import log, sqrt, exp
from scripts.gpt.errors import wax_gt, aoa_gt

random.seed(2022)
np.random.seed(515)
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# def chernoff_hoeffding_pval(mu, t, n):
#     x1 = (mu / (mu + t)) ** (mu + t)
#     x2 = ((1 - mu) / (1 - mu - t)) ** (1 - mu - t)
#     x = x1 * x2
#     return x ** n

def hoeffding_pval(t, n):
    return exp(-2 * (t ** 2) * n)

def hoeffding_twoside_error(n, confidence):
    delta = (1 - confidence) / 2
    return sqrt(1 / (2 * n) * log(1 / delta))

class HumanSimulator:

    def __init__(self, mu, rho=0):
        self.mu = mu
        self.rho = rho

    def sample(self, n=1, gpt_correct=None):
        if gpt_correct is not None and self.rho > 0:
            gpt_mu =  gpt_correct.sum() / gpt_correct.shape[0]
            new_mu = (self.mu - gpt_mu * self.rho) / (1 - self.rho)
            if new_mu < 1 and new_mu < 0:
                print(new_mu)
                print(gpt_mu)
                print(self.mu)
                print(self.rho)
                raise AttributeError('Human not possibled to simulate.')
            answers = []
            for gpt_answer in gpt_correct:
                if random.random() < self.rho:
                    answers.append(gpt_answer)
                else:
                    answer = int(random.random() < new_mu)
                    answers.append(answer)
            return np.array(answers)
        else:
            return np.random.binomial(1, self.mu, size=n)

def sim_td_test(correct, mu, n, rho=0, binom=False):
    h1 = HumanSimulator(mu, rho).sample(n, correct)
    h2 = HumanSimulator(mu, rho).sample(n, correct)
    h3 = HumanSimulator(mu, rho).sample(n, correct)
    td = np.abs(correct - h1).mean()
    # estimate td from gamma
    gamma = np.abs(h2 - h3).mean()
    if binom:
        return binom_test(n * td, n, gamma, alternative='greater'), td, gamma
    else: # use hoeffding's inequality
        # still one-sided, for this test, two-sided is times 4
        eps = td - gamma
        if eps > 0:
            p = hoeffding_pval(eps, n)
            # account for gamma being estimated from data
            # p = 2 * hoeffding_pval(eps / 2, n)
        else:
            p = 1. # fail to reject?
        if p > 1.: p = 1.
        return p, td, gamma

def plot_simulation(df, age_col, mu, fname):
    ages = list(set(df[age_col]))
    rhos = [0, .2, .25, .3]

    fig, ax = plt.subplots(ncols=len(rhos) + 2, nrows=1, figsize=(15,2.2), sharey=True, sharex=True)
    i = 0
    for binom in [True]:

        for rho in rhos + [0,0]:

            td_ages = []
            td_p_vals = []
            td_ph_vals = []
            bin_p_vals = []
            bin_ph_vals = []
            bin_ages = []

            for _ in range(25):

                for age in ages:

                    idx = df[age_col] <= age
                    # idx = df[age_col] == age
                    n = idx.sum()
                    correct = (df[idx]['gt'] == df[idx]['intent']).astype(int).values
                    h_correct = HumanSimulator(mu).sample(n)
                    p = sim_td_test(correct, mu, n, rho=rho, binom=binom)[0]
                    td_p_vals.append(p)
                    ph = sim_td_test(h_correct, mu, n, rho=rho, binom=binom)[0]
                    td_ph_vals.append(ph)
                    td_ages.append(age)

                    # binom test, only keeps last one to plot
                    if i == len(rhos):
                        bin_p = binom_test(correct.sum(), n, p=mu, alternative='less')
                        bin_p_vals.append(bin_p)
                        bin_ph = binom_test(h_correct.sum(), n, p=mu, alternative='less')
                        bin_ph_vals.append(bin_ph)
                        bin_ages.append(age)

                    if i == len(rhos) + 1:
                        bin_p = binom_test(correct.sum(), n, p=mu, alternative='less')
                        bin_p_vals.append(bin_p)
                        h_correct = HumanSimulator(mu, 0.3).sample(n, correct)
                        bin_ph = binom_test(h_correct.sum(), n, p=mu, alternative='less')
                        bin_ph_vals.append(bin_ph)
                        bin_ages.append(age)

            if i >= len(rhos):
                data = pd.DataFrame({
                    'p val' : bin_p_vals + bin_ph_vals, 
                    'ages' : bin_ages + bin_ages,
                    'shift' : ['GPT v. H'] * len(bin_p_vals) + ['H v. H'] * len(bin_ph_vals)})
                use_legend = False if (i != len(rhos) - 1) else 'full'
                ax.flat[i].axhline(0.05, c='r', ls='--')
                sns.lineplot(x='ages', y='p val', hue='shift', markers=['o', 'x'],
                    data=data, ax=ax.flat[i], legend=use_legend)
                if use_legend:
                    handles, labels = ax.flat[i].get_legend_handles_labels()
                    ax.flat[i].legend(handles=handles[1:], labels=labels[1:])
            #         ax.flat[i].plot(ages, td_p_vals, label=f'GPT v. H')
            #         ax.flat[i].plot(ages, td_ph_vals, label='H v. H')
                ax.flat[i].set_title(f'Mean Error Test\nagreement (rho) = 0', fontweight="bold")
                # if i <= len(rhos):
                #     ax.flat[i].set_xlabel(None)
                i += 1
                continue

            data = pd.DataFrame({
                'p val' : td_p_vals + td_ph_vals, 
                'ages' : td_ages + td_ages,
                'shift' : ['GPT v. H'] * len(td_p_vals) + ['H v. H'] * len(td_ph_vals)})
            with open('results/pvals.txt', 'a') as out:
                out.write('=======================================================\n')
                out.write(f'binom = {binom} rho = {rho:.2f} fail to reject age = a\n')
                summ = data.groupby(['ages', 'shift'])['p val'].mean()
                out.write(f'{summ[summ > 0.01].to_string()}\n')
            use_legend = False if (i != len(rhos) - 1) else 'full'
            ax.flat[i].axhline(0.05, c='r', ls='--')
            sns.lineplot(x='ages', y='p val', hue='shift', markers=['o', 'x'],
                data=data, ax=ax.flat[i], legend=use_legend)
            if use_legend:
                handles, labels = ax.flat[i].get_legend_handles_labels()
                ax.flat[i].legend(handles=handles[1:], labels=labels[1:])
        #         ax.flat[i].plot(ages, td_p_vals, label=f'GPT v. H')
        #         ax.flat[i].plot(ages, td_ph_vals, label='H v. H')
            if binom: t = 'TD Test' 
            else: t = 'TD Test (Hoeffding)'
            ax.flat[i].set_title(f'{t}\nagreement (rho) = {rho:.2f}', fontweight="bold")
            # if i <= len(rhos):
            #     ax.flat[i].set_xlabel(None)
            i += 1
        #         ax.flat[i].set_xlabel('Age')
        #         if i == 0:
        #             ax.flat[i].set_ylabel('p value')


    # plot binom test, only kept last
    i -= 1
    # ax.flat[i].axhline(0.05, c='r', ls='--')
    # bin_data = pd.DataFrame({
    #     'p val' : bin_p_vals + bin_ph_vals, 
    #     'ages' : bin_ages + bin_ages,
    #     'shift' : ['GPT v. H'] * len(bin_p_vals) + ['H v. H'] * len(bin_ph_vals)})
    # sns.lineplot(x='ages', y='p val', hue='shift', markers=['o', 'x'],
    #     data=bin_data, ax=ax.flat[i], legend=False)
    # ax.flat[len(rhos)].plot(ages, bin_p_vals, label='GPT v. H')
    # ax.flat[len(rhos)].plot(ages, bin_ph_vals, label='H v. H')
    # ax.flat[len(rhos)].set_xlabel('Age')
    ax.flat[i].set_title(f'Mean Error Test\nagreement (rho) = 0.30', fontweight="bold")
    # ax.flat[0].legend()
    # ax.flat[len(rhos)].legend()
    # if fname == 'wax':
    #     fig.suptitle(f'statistical tests for differences: word assoc.')
    # else:
    #     fig.suptitle(f'statistical tests for differences: word def.')
    plt.tight_layout(rect=[-0.03, -0.17, 1.03, 1.17])
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(f'results/{fname}-pvals')



if __name__ == '__main__':

    with open('results/pvals.txt', 'w') as out:
        pass
    wax = pd.read_csv('results/wax-interpreted.csv')
    wax['gt'] = wax.apply(wax_gt, axis=1)
    herr = hoeffding_twoside_error(110, .95)
    plot_simulation(wax, 'max_aoa_pair', .83 - herr, 'wax')
    # plot_simulation(wax, 'max_aoa_pair', .83 - herr, 'wax', binom=True)

    # aoa = pd.read_csv('results/aoa-interpreted.csv')
    # aoa['gt'] = aoa.apply(aoa_gt, axis=1)
    # plot_simulation(aoa, 'aoa', .735, 'aoa')