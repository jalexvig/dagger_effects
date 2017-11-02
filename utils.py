import pickle

import matplotlib.pyplot as plt


def plot(df1, df2, name=None, label1=None, label2=None):

    plt.errorbar(df1.index, df1.mean(axis=1), yerr=df1.std(axis=1), fmt='o', label=label1)
    plt.errorbar(df2.index, df2.mean(axis=1), yerr=df2.std(axis=1), fmt='o', label=label2)

    plt.xlabel('Num Observations')
    plt.ylabel(name.capitalize())

    plt.legend(loc=2)

    plt.show()


if __name__ == '__main__':

    with open('models/stats_hopper.pkl', 'rb') as f:
        s1, r1 = pickle.load(f)

    with open('models/stats_hopper_dagger.pkl', 'rb') as f:
        s2, r2 = pickle.load(f)

    plot(r1, r2, name='Reward', label1='Baseline', label2='Dagger')
    plot(s1, s2, name='Steps', label1='Baseline', label2='Dagger')
