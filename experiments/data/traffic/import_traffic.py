'''
From Ben Taieb KDD 2019: "The dataset gives the occupancy rate (between 0 and 1) of 963 car lanes of San Francisco
bay area freeways. The measurements are sampled every 10 minutes fromJan. 1st 2008 to Mar. 30th 2009. This dataset
has been notably usedin [ 4] for classi￿cation tasks. We aggregate the data to 366 daily observations split in120,
120and126observations for training,validation and testing, respectively.

We consider hierarchies with m=200 and k=7 (k=num of aggregated series), where each hierarchy is constructed as
follows. We sample 200 bottom level series from the 963 series, and computethe upper level series by aggregation.
More speci￿cally, 200 series at the bottom level were aggregated in groups of 50 for the nextlevel, resulting in 4
series. These 4 series were then aggregated in groups of two to obtain two aggregate series and the top level series."

Data downloaded from: https://archive.ics.uci.edu/ml/datasets/PEMS-SF

We use the following files:
-PEMS_train
-PEMS_test
-PEMS_trainlabels
-PEMS_testlabels
-randperm

The following files are unsed:
-stations_list

Comments:
- PEMS_train has 267 observations (this is mistakenly noted to be 263 on the website)
- PEMS_test has 173 observations
- Since the data are occupancy rates, to aggregate to each day, it would make most sense to average (a rate must be
in [0,1]). However, given the values suggested in Figure 3b of the Ben Taieb KDD, it seems that he simply sums them.
We will do the same, although I think it is incorrect.
- We invert the permutation as suggested in the description in the download link. It looks from Figure 3b that Ben
Taieb does the same thing, but he doesn't report it
- The scheme described by Ben Taieb results in a 4-level hierarchy
- We take the first 366 observations to get a full year. Note that this is not actually a year because there was one
day removed due to anomalies and public holidays were also taken out by dataset author.
- Ben Taieb explains in Section 5 paragraph 1 of KDD paper that they run this sampling 100 times, so effectively 100
datasets. Do we want to do this?
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools

##  Helper functions
def process_line(l):
    '''
    INPUTS:
    l - string

    OUTPUTS:
    l - list
    '''

    l = l.replace(';', ' ').replace('[', '').replace(']', '').replace('\n', '')

    assert l.count(';') == 0 and l.count('[') == 0 and l.count(']') == 0 and l.count('\n') == 0

    l = l.split(' ')
    l = [float(i) for i in l]
    assert len(l) == 963*144
    return l

def aggregate_day(d, method='sum'):
    '''
    Each input is a day of data. It will have 963*144 observations. We want to /sum/average over each day, so break list
    up into chunks of 144
    INPUTS:
    d - list of length 963*144
    method - string that indicates whether observations are averaged or summed over each day. Ben Taieb KDD 2019 uses
    sum
    
    OUTPUTS:
    list of length 963
    '''

    num_obs = len(d)
    assert num_obs == 963*144
    n = 144 # observations every 10 mins, so a total of 6*24 = 144 per day
    agg = [sum(d[i:i+n]) for i in range(0, num_obs, n)]

    if method == 'sum':
        return agg
    else:
        return [e / n for e in agg]

def invert_permutation(perm):
    '''
    Inverts a permutation list. Returns a new permutation list.
    '''
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

## Main function
def import_traffic(dirname, alldata=False, plotting=True, randseed=13):

    # 'dirname' must have trailing backslash, i.e., /traffic_data/
    # Import raw data and labels. They are lists of length 267 (train) and 173 (test), each sublist has length 963*144
    f_train = open(dirname + 'PEMS_train','r')
    f_test = open(dirname + 'PEMS_test','r')
    raw_data_train = [process_line(l) for l in f_train.readlines()]
    raw_data_test = [process_line(l) for l in f_test.readlines()]

    f_train = open(dirname + 'PEMS_trainlabels','r') # reuse f_train
    f_test = open(dirname + 'PEMS_testlabels','r') # reuse f_test
    raw_labels_train = [int(e) for e in f_train.readlines()[0].replace('[', '').replace(']', '').split(' ')]
    raw_labels_test = [int(e) for e in f_test.readlines()[0].replace('[', '').replace(']', '').split(' ')]

    # Append test to train data to get a lists of 267+173=440 observations
    raw_data = raw_data_train + raw_data_test
    raw_labels = raw_labels_train + raw_labels_test

    # Aggregate for each day (list of length 440, each sublist of length 963)
    agg_data = [aggregate_day(d) for d in raw_data]

    # Get permutation and re-order days in agg_data
    f_perm = open(dirname + 'randperm','r')
    perm_list = [int(e) - 1 for e in f_perm.readlines()[0].replace('[', '').replace(']', '').split(' ')] #list of length 440, convert index to 0,1,2,...

    agg_data = [agg_data[i] for i in invert_permutation(perm_list)]
    labels = [raw_labels[i] for i in invert_permutation(perm_list)] #check this to see that permutation is done correctly. Otherwise we don't need it further.

    # Convert to numpy
    data = np.array(agg_data).T

    # Following Ben Taieb, we take only the (1st?) 366 days (i.e., 1 year). The flag alldata will default to False
    if not alldata:
        data = data[:,:366]

    # Sample 200 bottom series
    num_samples = 200
    np.random.seed(randseed) #fix random seed
    rand_idxs = np.random.choice(data.shape[0], size=num_samples, replace=False)
    data_200 = data[rand_idxs, :]

    # Make bottom series labels
    bottom_labels = ['Bottom' + str(i + 1) for i in range(data_200.shape[0])]

    # Create level dictionary with labels and information about hierarchy
    level_dict = {
        'Total': ['Total'],
        'Two': ['y1','y2'],
        'Four': ['y11','y12','y21','y22'],
        'Bottom':bottom_labels
    }

    all_labels = list(itertools.chain(*level_dict.values()))
    num_ts = len(all_labels)
    num_bottom_ts = len(bottom_labels)
    num_agg_ts = num_ts - num_bottom_ts

    # Create S matrix manually
    S = np.zeros((num_ts, num_bottom_ts))
    S[0,:] = 1
    S[1,:int(num_samples / 2)] = 1
    S[2,-int(num_samples / 2):] = 1
    S[3,:int(num_samples / 4)] = 1
    S[4,int(num_samples / 4):int(num_samples / 2)] = 1
    S[5,int(num_samples / 2):int(3 * num_samples / 4)] = 1
    S[6,-int(num_samples / 4):] = 1
    S[-num_samples:,:] = np.eye(num_samples)

    # Generate Y matrix
    Y = S @ data_200

    # Plotting if True
    if plotting:
        # Plot S matrix
        plt.figure(num=1, figsize=(8, 10), dpi=100, facecolor='w')
        plt.spy(S)
        plt.show()

        # Plot Y data
        # plot 30 time series from total of 207. Plot all 7 aggregate and select 23 random others
        num_plots = 30
        np.random.seed(randseed + 2)
        plot_bottom_idxs = np.random.choice(num_samples, num_plots-num_agg_ts, replace=False)

        fig, axs = plt.subplots(10, 3, num=2, figsize=(20, 15), facecolor='w')
        fig.subplots_adjust(hspace=.6, wspace=.001)

        # Set font sizes
        small = 6
        med = 8
        big = 10
        plt.rc('font', size=small)  # controls default text sizes
        plt.rc('axes', titlesize=small)  # fontsize of the axes title
        plt.rc('axes', labelsize=small)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
        plt.rc('legend', fontsize=small)  # legend fontsize
        plt.rc('figure', titlesize=big)  # fontsize of the figure title

        axs = axs.ravel()

        # Do aggregate 7
        for i in range(num_agg_ts):
            axs[i].plot(np.arange(0, Y.shape[1], 1), Y[i, :])
            axs[i].set_title(all_labels[i])

        # Do remaining random 23
        for i in range(num_agg_ts, num_plots):
            ts_idx = plot_bottom_idxs[i - num_agg_ts] + num_agg_ts
            axs[i].plot(np.arange(0, Y.shape[1], 1), Y[ts_idx, :])
            axs[i].set_title(all_labels[ts_idx])

        plt.show()

    # Save data to csv

    # Indices and timestamps
    start_date = '2008-01-01'  # should be same for all series
    date_index = pd.date_range(
        start=start_date, periods=Y.shape[1], freq="D"
    )

    # Y data
    dataY = {
        str(column): Y[i, :]
        for i, column in enumerate(all_labels)
    }
    df = pd.DataFrame(
        index=date_index,
        data=dataY,
    )
    df.to_csv('./data.csv')

    # sanity check for Y
    data = pd.read_csv('./data.csv', index_col=0)
    values = data.values.transpose()
    assert np.max(np.abs(Y - values)) <= 1e-9  #
    # assert data.index.equals(index)

    # S matrix
    dataS = {
        str(column): S[:, i]
        for i, column in enumerate(bottom_labels)
    }
    agg_mat_df = pd.DataFrame(
        index=all_labels,
        data=dataS
    )

    agg_mat_df.to_csv('./agg_mat.csv')

    # sanity check for S
    agg_mat = pd.read_csv('./agg_mat.csv', index_col=0).values
    assert (agg_mat == S).all()

    return print('Importing UCI Traffic Data successful!...')

if __name__ == "__main__":
    import_traffic('traffic_data/')