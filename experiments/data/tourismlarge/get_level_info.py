import pandas as pd


row_labels = pd.read_csv(f'./agg_mat.csv').iloc[:, 0].values

change_ix = [0, 1]
cur_len = len(row_labels[1])
for i in range(2, len(row_labels)):
    if len(row_labels[i]) != cur_len:
        change_ix.append(i)
        cur_len = len(row_labels[i])

print(change_ix)
