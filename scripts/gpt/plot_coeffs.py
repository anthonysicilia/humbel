import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')

plt.rcParams.update({'font.size': 22})

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Intercept              0.3302      0.045      7.402      0.000       0.243       0.418
# adx[T.ADX]             0.0349      0.011      3.129      0.002       0.013       0.057
# xy[T.X Y]              0.0311      0.011      2.849      0.004       0.010       0.052
# relation[T.hard]       0.1095      0.039      2.845      0.004       0.034       0.185
# num_morphs[T.high]     0.0228      0.017      1.376      0.169      -0.010       0.055
# expl[T.True]           0.0615      0.010      6.226      0.000       0.042       0.081
# max_aoa_pair           0.0044      0.001      3.056      0.002       0.002       0.007

df = pd.DataFrame({
    'Hypothesis' : ['H1: POS=Adv,Adj', 'H2: POS X!=Y', 'H3: Hard Rel.', 'H4: High Morph.C', 'H5: GPT Explains', 'H6: +1 Incr. AoA'],
    'Effect Size' : [0.0349, 0.0311, 0.1095, 0.0228, 0.0615, 0.0044]})

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6,2.3))
    sns.barplot(data=df, x="Hypothesis", y="Effect Size")
    plt.title('Expected Increase in Probability of Error for Each Event', fontweight="bold")
    plt.xlabel(None)
    plt.xticks(rotation=15)
    plt.tight_layout(rect=[-0.03, -0.1, 1.03, 1.1])
    plt.savefig('results/coeffs')
    