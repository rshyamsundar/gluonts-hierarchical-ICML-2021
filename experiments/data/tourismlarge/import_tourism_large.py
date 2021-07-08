'''
Author: Pedro, Lucien
Dec 10, 2020

Data and paper from here: https://robjhyndman.com/publications/mint/
Frequency is monthly, total of 36 observations


Description:
'''

# Imports
import pandas as pd
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

def import_tourism_large(filename, plotting=False):

    # filename = 'hyndman_tourism_large.csv'
    df = pd.read_csv(filename)

    bottom_ts_labels = list(df.columns)[2:] # ignore data columns
    num_bottom_ts = len(bottom_ts_labels)

    num_levels = len(bottom_ts_labels[0]) - 1 # add 1 for the root node 'Total'

    # Get labels for all series including aggregate
    level_labels = ['Total','State','Zone','Region','Type'] # order is important here! Taken from Table 6 in https://robjhyndman.com/papers/mint.pdf
    level_dict = OrderedDict()
    for idx,l in enumerate(level_labels):
        if l == 'Total':
            level_dict[l] = ('Total',)
        else:
            letter_list = set([e[idx-1] for e in bottom_ts_labels]) # use set to get unique
            level_dict[l] = sorted(list(letter_list))


    # Should be 555 series (yay!)
    purpose = ['All', 'Hol', 'Vis', 'Bus', 'Oth']
    all_labels = []
    for i in range(num_levels-1):
        level_list = []
        if i == 0:
            for p in purpose:
                all_labels.append('Total' + p)
        else:
            for l in bottom_ts_labels:
                level_list.append(l[:i])
            # Get unique and sort
            level_list = sorted(list(set(level_list)))

            # Append type to each element of the list
            all_labels.extend([e + p for e in level_list for p in purpose])

    num_ts = len(all_labels)

    S = np.zeros((num_ts, num_bottom_ts))

    row = 0 #row counter
    # Remaining rows loop through all labels
    for l in all_labels:
        #get 1st part of label (State/Region/Zone)
        part1 = l[:-3]
        #get 2nd part of label (purpose)
        part2 = l[-3:]

        # Totals
        if part1 == 'Total':
            if part2 == 'All':
                mask = [1 for bl in bottom_ts_labels]
            else:
                mask = [1 if (bl.endswith(part2)) else 0 for bl in bottom_ts_labels]

        # All others
        else:
            if part2 == 'All':
                mask = [1 if bl.startswith(part1) else 0 for bl in bottom_ts_labels]
            else:
                mask = [1 if (bl.startswith(part1) and bl.endswith(part2)) else 0 for bl in bottom_ts_labels]

        S[row,:] = mask
        row += 1

    # Get bottom series Y data
    Y_bottom = df.transpose()[2:][:].to_numpy()

    # Compute all Y data using S
    Y = S @ Y_bottom

    # Plotting if True
    if plotting:
        # Plot S matrix
        plt.figure(num=1, figsize=(8, 20), dpi=80, facecolor='w')
        plt.spy(S)
        plt.show()

        # Plot Y data (only the first 60 of 555)
        # Get shortened plot titles
        acronyms = []
        for i in all_labels:
            if i == []:
                acronyms.append(['Total'])
            else:
                ac = ["".join(e[0] for e in j.split()).upper() for j in i]
                acronyms.append(ac)
        fig, axs = plt.subplots(6,10, num = 2, figsize=(28, 14), facecolor='w')
        fig.subplots_adjust(hspace = .5, wspace=.001)

        # Set font sizes
        small = 7
        med = 10
        big = 12
        plt.rc('font', size=small)          # controls default text sizes
        plt.rc('axes', titlesize=med)     # fontsize of the axes title
        plt.rc('axes', labelsize=small)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
        plt.rc('legend', fontsize=small)    # legend fontsize
        plt.rc('figure', titlesize=big)  # fontsize of the figure title

        axs = axs.ravel()
        for i in range(6*10):
            axs[i].plot(np.arange(0,Y.shape[1],1),Y[i,:])
            axs[i].set_title(all_labels[i])

        plt.show()

    # Save data to csv

    # Indices and timestamps
    index = pd.date_range(
        start=pd.Timestamp('1998-01-01'), periods=Y.shape[1], freq="MS"
    )

    # Y data

    dataY = {
        str(column): Y[i, :]
        for i, column in enumerate(all_labels)
    }
    df = pd.DataFrame(
        index=index,
        data=dataY,
    )
    df.to_csv('./data.csv')

    # sanity check for Y
    data = pd.read_csv('./data.csv', index_col=0)
    values = data.values.transpose()
    assert np.max(np.abs(Y - values)) <= 1e-6  # values in this dataset are large
    # assert data.index.equals(index)

    # S matrix
    dataS={
            str(column): S[:, i]
            for i,column in enumerate(bottom_ts_labels)
    }
    agg_mat_df = pd.DataFrame(
        index=[str(i) for i in all_labels],
        data=dataS
    )

    agg_mat_df.to_csv('./agg_mat.csv')

    # sanity check for S
    agg_mat = pd.read_csv('./agg_mat.csv', index_col=0).values
    assert (agg_mat == S).all()

    return print('Importing ' + filename + ' successful!...')

if __name__ == "__main__":
    import_tourism_large('hyndman_tourism_large.csv')

