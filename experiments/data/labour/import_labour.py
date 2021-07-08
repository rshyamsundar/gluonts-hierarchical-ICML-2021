'''
Author: Lucien
Jan 15, 2021

Data is Table 12 from here: https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release#data-download
Original file name should be '6YEAR12.xls' (e.g., '6202012.xls'). Renamed to 'australian_labour.xls'

Paper where data is referenced is the SHARQ submission to Aistats

Description: ""
'''

'''

'''

import pandas as pd
import numpy as np
import itertools
from collections import OrderedDict
from matplotlib import pyplot as plt

## Helper functions
def is_valid_tuple(mylist, lists):
    assert len(mylist) == len(lists)
    visited = [False] * len(mylist)
    for e in mylist:
        for idx,l in enumerate(lists):
            if e in l:
                visited[idx] = True
                break
    if all(visited):
        return True
    else:
        return False

## Main function
def import_labour(filename, plotting=False):
    # filename = 'australian_labour.xls'

    # Import 3 sheets from excel file and concatenate
    xls = pd.ExcelFile(filename)
    df_list = []
    for i in range(3):
        sheet_name = 'Data' + str(i+1)
        df_list.append(pd.read_excel(xls, sheet_name, header=None))
    df = pd.concat(df_list, ignore_index=True, axis=1).transpose()

    # Delete all series that are trend and seasonally adjusted. Only keep those labeled "Original" in Series Type (3rd column)
    df = df[df[2] == 'Original'] #keeps 378 series

    # Delete all series (rows) starting with "Employment to population ratio", "Unemployment rate", "Participation ratio".
    # These categories are cannot be incorporated into a hierarchy
    df = df[~df[0].astype(str).str.startswith('Employment to population ratio')] #75 series
    df = df[~df[0].astype(str).str.startswith('Unemployment rate')] #75 series
    df = df[~df[0].astype(str).str.startswith('Participation rate')] #75 series

    # After these removals, total of 297 series should be left (verify this with the xls)

    # Delete unneeded columns and reset index
    df = df.drop(df.columns[[2, 3, 5,8]], axis=1)
    df.columns = range(df.shape[1])

    # Split first row on semicolon
    df[0]= df[0].str.split(";|>", expand = False)

    # Strip empty spaces at beg and end of each string in the list of strings in df[0].
    # Also remove empty strings from list with filter()
    def strip_spaces(my_list):
        return list(filter(None, [x.rstrip().lstrip() for x in my_list]))

    df[0] = df[0].apply(strip_spaces)

    # Dictionary with all feature categories and members of each category. This is for the entire dataset but is not complete here yet
    feature_dict = {'Total': ['Total'],
                    'Location': ['New South Wales','Victoria','Queensland','South Australia','Western Australia','Tasmania','Northern Territory','Australian Capital Territory'], #aggregate feature is 'Australia'
                    'Gender': ['Males','Females'], # aggregate feature is 'Persons'
                    'Status': ['Employed full-time','Employed part-time'] # aggregate features is 'Employed Persons'
                    }

    # Get hierarhcy from SHARQ paper
    '''
    Section D.4 in the Aistats paper states the following 
    "
    Specifically, the 32 bottom level series are hierarchically aggregated using labour force location, gender and employment status.
    "
    This suggests the following levels in the tree:
    {1: Total, (1 options)
     2: Location, (8 options)
     3: Gender, (2 options)
     4: Status, (2 options)
    } 
    These values are the categories taken from the feature_dict
    There will be 32 bottom series and 25 aggregate series.
    '''
    sub_categories = ['Location', 'Gender', 'Status'] #order here is important!! It determines the hierarchy
    sub_feature_dict = dict((key, feature_dict[key]) for key in sub_categories if key in feature_dict)

    #filter dictionary
    df = df[df[0].apply(is_valid_tuple, lists = list(sub_feature_dict.values()))]

    # Reset row indices to sequential values
    df.reset_index(drop=True, inplace=True) #reset indexes

    # Get labels for all series including aggregate
    level_labels = sub_categories
    level_labels.insert(0,'Total')
    level_dict = OrderedDict()
    for l in level_labels:
        if l == 'Total':
            level_dict[l] = ('Total',)
        else:
            level_dict[l] = tuple(sub_feature_dict[l])

    bottom_ts_labels = list(df.iloc[:, 0]) # bottom ts labels
    num_bottom_ts = len(bottom_ts_labels)

    level_list = bottom_ts_labels.copy()
    level_labels_dict = level_dict.copy()
    for k, v in reversed(level_dict.items()):
        level_labels_dict[k] = level_list
        temp_list = []
        for l in level_list:
            temp_list.append([x for x in l if x not in v])
        temp_list.sort()
        level_list = list(k for k, _ in itertools.groupby(temp_list))

    all_labels = list(itertools.chain(*level_labels_dict.values()))

    # Define S matrix
    S = np.zeros((len(all_labels), num_bottom_ts))
    for idx,l in enumerate(all_labels):
        if l == []: # first row
            S[idx, :] = 1
        else:
            for jdx,t in enumerate(bottom_ts_labels):
                if all(elem in t  for elem in l):
                    S[idx,jdx] = 1


    # Get bottom series Y data
    Y_bottom = df.transpose()[6:][:].to_numpy().T

    # Compute all Y data using S
    Y = S @ Y_bottom

    # Plotting if True
    if plotting:
        # Plot S matrix
        plt.figure(num=1, figsize=(8, 20), dpi=80, facecolor='w')
        plt.spy(S)
        plt.show()

        # Plot Y data
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
        for i in range(len(all_labels)):
            axs[i].plot(np.arange(0,Y.shape[1],1),Y[i,:])
            axs[i].set_title(str(acronyms[i]))

        # Delete empty subplots
        for i in range(57,60):
            fig.delaxes(axs[i])
        plt.show()

    # Save data to csv

    # Indices and timestamps
    start_date = df.iloc[0,3] #should be same for all series
    index = pd.date_range(
        start=start_date, periods=Y.shape[1], freq="MS"
    )

    # Y data
    # Deal with first label in ts labels list
    all_labels[0] = ['Total']

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
    dataS = {
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
    import_labour('australian_labour.xls')
