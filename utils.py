import pandas as pd
import numpy as np
train_dataset = pd.read_csv('train.csv', parse_dates=['epoch'], index_col='id')
test_dataset = pd.read_csv('test.csv', parse_dates=['epoch'], index_col='id')
train_dataset['is_train'] = True
test_dataset['is_train'] = False

train_dataset['millis'] = train_dataset.epoch.astype(np.int64)
test_dataset['millis'] = test_dataset.epoch.astype(np.int64)
dataset = pd.concat([train_dataset, test_dataset])

gt_position = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
sim_position = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']


def without_duplicates(df):
    bad_ids = []
    copy_data = {}
    for sat_id in np.unique(df.sat_id.values):
        sat = df[df.sat_id == sat_id]
        sat['time_delta'] = sat.millis.shift(-1) - sat.millis
        time = np.quantile(sat['time_delta'][:-1].values, 0.1) / 2
        remove = sat[sat['time_delta'].map(lambda x: time > x)].index.values
        bad_ids.extend(remove)
        copy_from = sat[sat['time_delta'].shift(1).map(lambda x: time > x)].index.values
        for a, b in zip(copy_from, remove):
            copy_data[a] = b
    return df[~df.index.isin(bad_ids)], copy_data






















