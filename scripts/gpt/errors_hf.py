from scripts.gpt.errors import wax_gt
from scipy.stats import binom_test
import pandas as pd

if __name__ == '__main__':

    leq = lambda x,y: x <= y
    eq = lambda x,y: x == y
    # wax_gamma = 0.66
    wax_gamma = 0.47

    # results_dir = 'results-meta-llama+Llama-2-7b-chat-hf'
    # results_dir = 'results-HuggingFaceH4+zephyr-7b-beta'
    # results_dir = 'results-mistralai+Mistral-7B-Instruct-v0.2'
    # results_dir = 'results'
    # results_dir = 'results-chat'
    # results_dir = 'results-chat-new'
    results_dir = 'results-chat-icl-10'

    for s, comp in [('= AoA', eq)]:#[('$\leq$AoA', leq), ('= AoA', eq)]:

        wax = pd.read_csv(f'{results_dir}/wax-interpreted.csv')

        wax_errors = []
        wax_ages = []

        wax['gt'] = wax.apply(wax_gt, axis=1)
        for age in set(wax['max_aoa_pair']):
            idx = comp(wax['max_aoa_pair'], age)
            # idx = wax['max_aoa_pair'] == age
            correct = (wax[idx]['gt'] == wax[idx]['intent']).sum()
            p = binom_test(correct, idx.sum(), wax_gamma, alternative='less')
            correct = correct / idx.sum()
            print(f'\% correct wax at {age}: {100 * correct : .1f} (n={idx.sum()}) (p={p:.2f})')
            wax_errors.append(100 * correct)
            wax_ages.append(age)
        correct = (wax['gt'] == wax['intent']).sum() / len(wax)
        print(f'\% correct wax overall: {correct}')