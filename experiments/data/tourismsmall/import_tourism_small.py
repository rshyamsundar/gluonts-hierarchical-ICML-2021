'''
Author: Pedro, Lucien
Dec 10, 2020

Data from here: https://robjhyndman.com/publications/hierarchical-tourism/
Frequency is monthly, total of 36 observations

Paper on the data here: https://robjhyndman.com/papers/hiertourism.pdf

Description: "For each domestic tourismsmall demand time series, we have quarterly observations on the number ofvisitor
nights which we use as an indicator of tourismsmall activity.  The sample begins with the firstquarter of 1998 and ends
with the final quarter of 2006."
'''

# Imports
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

def import_tourism_small(filename, plotting=False):
    #filename = 'hyndman_tourism_small.csv'
    df = pd.read_csv(filename)

    ts_labels = list(df.columns)
    ts_labels = [ts_label.lower().replace(" ", "") for ts_label in ts_labels]
    levels = defaultdict(list)
    for ts_label in ts_labels:
        info = ts_label.split('-')

        if len(info) == 1:
            if info[0] == 'total':
                levels[0].append(['total'])
            else:
                levels[1].append(ts_label)
        elif len(info) == 2:
            levels[2].append(ts_label)
        elif len(info) == 3:
            levels[3].append(ts_label)

    num_levels          = len(levels)
    bottom_level        = levels[num_levels - 1]
    num_bottom_level_ts = len(bottom_level)
    num_ts              = len(ts_labels)
    S = np.zeros((num_ts,num_bottom_level_ts))

    # set level zero row to one
    S[0, :] = 1

    # go for the remaining time series
    for idx_row,ts_label in enumerate(ts_labels):
        for idx_col,bottom_level_ts in enumerate(bottom_level):
            if ts_label in bottom_level_ts:
                S[idx_row,idx_col]=1

    #get data
    Y = df.to_numpy().T #shape = (89, 36) Only 36 datapoints in the set!

    if plotting:

        #plot S matrix
        plt.figure(num=1, figsize=(8, 20), dpi=80, facecolor='w', edgecolor='k')
        plt.spy(S)
        plt.show()

        #Plotting Y data
        plt.figure(num=2, dpi=100, facecolor='w', edgecolor='k')
        plt.stackplot(np.arange(0,36,1), Y[1:5,:])
        plt.plot(np.arange(0,36,1), Y[0,:], color = 'black', linewidth=2)
        plt.show()

    ## Save data into CSV format (same as sine7())
    # Indices and timestamps
    index = pd.date_range(
        start=pd.Timestamp("1998-01-01"), periods=Y.shape[1], freq="Q"
    )

    #Y data
    data = {
        column: Y[i, :]
        for i, column in enumerate(ts_labels)
    }
    df = pd.DataFrame(
        index=index,
        data=data,
    )

    df.to_csv('./data.csv')

    # sanity check
    data = pd.read_csv('./data.csv', index_col=0)
    values = data.values.transpose()
    assert np.max(np.abs(Y - values)) <= 1e-6 #values in this dataset are large
    #assert data.index.equals(index)

    #S matrix
    agg_mat_df = pd.DataFrame(
        index=ts_labels,
        data={
            bottom_level[i]: S[:, i]
            for i in range(num_bottom_level_ts)
        }
    )

    agg_mat_df.to_csv('./agg_mat.csv')

    # sanity check
    agg_mat = pd.read_csv('./agg_mat.csv', index_col=0).values
    assert (agg_mat == S).all()

    return print('Importing ' + filename + ' successful!...')

if __name__ == "__main__":
    import_tourism_small('hyndman_tourism_small.csv')
