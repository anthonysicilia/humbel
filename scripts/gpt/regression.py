import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scripts.gpt.morph import morph_cats
from scripts.gpt.pos import filter_fn as pos_filter_fn
from scripts.gpt.rel import filter_fn as rel_filter_fn

chat = True

def set_pos_baseline_dif(s):
    num = len(set(s.split(' ')))
    if num > 1:
        return 'X Y'
    else:
        return 'X X'

def set_pos_baseline_adx(s):
    if 'AD' in s:
        return 'ADX'
    else:
        return 'AA'

def set_rel_baseline(s):
    hard = {'Function', 'HasProperty', 'Result-In', 'UNK'}
    if s in hard:
        return 'hard'
    else:
        return 'AA'

def set_morph_baseline(s):
    if 'low' in s:
        return 'AA low'
    else:
        return 'high'
    
FN_NAME = 'results-chat/regression-out.txt' if chat \
    else 'results/regression-out.txt'

def write(*args):
    s = ' '.join([str(a) for a in args])
    with open(FN_NAME, 'a') as out:
        out.write(f'{s}\n')

if __name__ == '__main__':

    with open(FN_NAME, 'w') as out:
        pass

    wax = pd.read_csv('results-chat/wax-features.csv').dropna() if chat \
        else pd.read_csv('results/wax-features.csv').dropna()

    # pre-proc steps: pos
    wax = pos_filter_fn(wax)
    wax['adx'] = wax['pair_pos'].apply(set_pos_baseline_adx)
    wax['xy'] = wax['pair_pos'].apply(set_pos_baseline_dif)
    # pre-proc steps: rel
    wax = rel_filter_fn(wax)
    wax['relation'] = wax['relation'].apply(set_rel_baseline)
    # pre-proc steps: morph
    wax['num_morphs'] = wax['num_morphs'].map(morph_cats)
    wax['num_morphs'] = wax['num_morphs'].apply(set_morph_baseline)
    # pre-proc steps: exp
    wax['expl'] = wax['exp']

    # no interaction model
    formula = 'incorrect ~ max_aoa_pair'\
        ' + adx + xy + relation + num_morphs + expl'\
        # ' + max_aoa_pair * relation'\
        # ' + max_aoa_pair * adx'\
        # ' + max_aoa_pair * xy'\
        # ' + max_aoa_pair * num_morphs'

    lreg = smf.ols(formula, data=wax).fit(cov_type='HC0')
    write('Base Model')
    write(lreg.summary())
    write('Bigger than 1:', (lreg.predict() > 1).sum())
    write('Less than 0:', (lreg.predict() < 0).sum())

    # interactions model, not significant
    formula = 'incorrect ~ max_aoa_pair'\
        ' + adx + xy + relation + num_morphs + expl'\
        ' + max_aoa_pair * relation'\
        ' + max_aoa_pair * adx'\
        ' + max_aoa_pair * xy'\
        ' + max_aoa_pair * num_morphs'
    
    lreg = smf.ols(formula, data=wax).fit(cov_type='HC0')
    write('Interactions Model')
    write(lreg.summary())
    write('Bigger than 1:', (lreg.predict() > 1).sum())
    write('Less than 0:', (lreg.predict() < 0).sum())

    # logit, no interaction model
    formula = 'incorrect ~ max_aoa_pair'\
        ' + adx + xy + relation + num_morphs + expl'\
        # ' + max_aoa_pair * relation'\
        # ' + max_aoa_pair * adx'\
        # ' + max_aoa_pair * xy'\
        # ' + max_aoa_pair * num_morphs'

    lreg = smf.logit(formula, data=wax).fit()
    write('Logit Model')
    write(lreg.summary())