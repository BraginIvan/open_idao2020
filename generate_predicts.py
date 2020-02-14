import utils
from extrapolator import Extrapolator
import numpy as np
import pandas as pd
with_val = False

train_dataset, _ = utils.without_duplicates(utils.train_dataset)
if with_val:
    path = 'predicts_val'
    train_val_time = train_dataset.millis.mean()
    train_dataset = train_dataset[train_dataset.millis < train_val_time]
    sat_ids = train_dataset.sat_id.unique()
else:
    path = 'predicts'
    full_test_dataset = pd.read_csv('test.csv', parse_dates=['epoch'], index_col='id')
    sat_ids = full_test_dataset.sat_id.unique()


def generate(sat_id):

    train_sat_data = train_dataset[train_dataset.sat_id == sat_id]

    take_size = int(len(train_sat_data) * 1.3)
    print(sat_id, take_size)
    train_sat_data=train_sat_data.iloc[-1500:]

    extrapolator = Extrapolator(sat_id)

    extrapolator.train(train_sat_data)
    print("--------------------------------------")
    print(sat_id, extrapolator.found_parameters)
    print("--------------------------------------")
    val_predicts = extrapolator.eval(len(train_sat_data), take_size)
    np.save('./%s/%d' % (path, sat_id), val_predicts)


import multiprocessing
full_test_dataset = pd.read_csv('train.csv', parse_dates=['epoch'], index_col='id')
cores_n = 3
multiprocessing.Pool(cores_n).map(generate, sat_ids, chunksize=1) # takes ~48/cores_n hours

