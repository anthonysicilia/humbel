import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filter_rowwise_duplitcates(df, cols):
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i > j:
                dups = df[df[ci] == df[cj]].index
                df.drop(dups, inplace=True)

def append_randomized_test(df, gt, alt_col, n_alt, random_state=515):
    for i in range(n_alt):
        df[f'alt{i}'] = df[alt_col]\
            .sample(frac=1, random_state=random_state+i)\
            .reset_index(drop=True)
    test_cols = gt + [f'alt{i}' for i in range(n_alt)]
    randomized = df[test_cols].apply(np.random.permutation, axis=1)
    cols = [f'test{i}' for i in range(n_alt + len(gt))]
    df[cols] = pd.DataFrame(randomized.to_list(), columns=cols)
    filter_rowwise_duplitcates(df, cols)

if __name__ == '__main__':

    aoa_data = pd.read_csv('originals/aoa.csv')
    wax = pd.read_csv('originals/wax.csv')

    aoa = dict()
    AGE_ADJUSTMENT = 5
    
    for word, grade in zip(aoa_data['WORD'], aoa_data['AoAtestbased']):
        # simple transform to get typical age from grade
        age = grade + AGE_ADJUSTMENT
        if word in aoa:
            # when a word has multiple meanings, assume the later aoa
            aoa[word] = max(aoa[word], age)
        else:
            aoa[word] = age
    
    # create direct aoa test dataset
    aoa_test = pd.DataFrame()
    aoa_test['word'] = aoa_data['WORD']
    aoa_test['def'] = aoa_data['MEANING']
    aoa_test['aoa'] = aoa_data['AoAtestbased'] + AGE_ADJUSTMENT

    # appends aoa test data
    aoa_test_full = aoa_test.copy()
    append_randomized_test(aoa_test_full, 
        gt=['word'], 
        alt_col='word',
        n_alt=3)
    aoa_test_full.to_csv('tests/aoa-test-full.csv')

    # just extract wax words and save
    wax_subset = set(wax['cue'].unique())\
        .union(set(wax['association'].unique()))
    aoa_test = aoa_test[aoa_test['word'].isin(wax_subset)].reset_index(drop=True)
    append_randomized_test(aoa_test, 
        gt=['word'], 
        alt_col='word',
        n_alt=3)
    aoa_test.to_csv('tests/aoa-test.csv')
    
    # create triplets N=1 or quadruples N=2 or quintuples N=3 by randomly sampling
    N = 2
    # appends wax test data
    append_randomized_test(wax, 
        gt=['cue', 'association'], 
        alt_col='association',
        n_alt=N)

    # assign aoas and filters nulls
    wax['cue_aoa'] = wax['cue'].map(aoa)
    wax['assoc_aoa'] = wax['association'].map(aoa)
    for i in range(N+2):
        wax[f'test{i}_aoa'] = wax[f'test{i}'].map(aoa)
    for i in range(N+2):
        wax = wax[~wax[f'test{i}_aoa'].isna()]
    
    # filter ambiguous cue/assoc. alts
    seen = set()
    for c, a in zip(wax['cue'], wax['association']):
        seen.add(frozenset((c,a)))
    ambig = []
    test_cols = [f'test{i}' for i in range(N+2)]
    for idx in wax.index:
        matches = 0
        for i, ci in enumerate(test_cols):
            for j, cj in enumerate(test_cols):
                if i > j:
                    pair = (wax.at[idx, ci], wax.at[idx, cj])
                    pair = frozenset(pair)
                    matches += int(pair in seen)
        if matches > 1:
            ambig.append(idx)
    wax.drop(ambig, inplace=True)

    # aoa for word list is largest of all aoas
    wax['max_aoa_all'] = wax[[f'test{i}_aoa' for i in range(N+2)]].max(axis=1)
    wax['max_aoa_pair'] = wax[['cue_aoa', 'assoc_aoa']].max(axis=1)
    wax.to_csv('tests/wax-test.csv')

    wax.sample(frac=0.01, random_state=515).to_csv('tests/wax-test-sample.csv')

    # plot data
    plt.hist(wax['max_aoa_all'], bins=20)
    plt.title('Max AoA of Pair + Alts')
    plt.savefig('tests/pair+alts-aoa')
    plt.clf()
    plt.hist(wax['max_aoa_pair'], bins=20)
    plt.title('Max AoA of Pair')
    plt.savefig('tests/pair-aoa')
    word_aoas = list(wax['cue'].drop_duplicates().map(aoa))
    word_aoas += list(wax['association'].drop_duplicates().map(aoa))
    plt.clf()
    plt.hist(word_aoas, bins=20)
    plt.title('AoA of Individual Words')
    plt.savefig('tests/word-aoa')

    
    print(aoa_test)
    print(wax)
    # print(wax['cue'].value_counts().sort_values())
    