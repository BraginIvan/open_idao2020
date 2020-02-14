import numpy as np
import pandas as pd

full_train_dataset = pd.read_csv('train.csv')
full_test_dataset = pd.read_csv('test.csv')
ids = set(full_train_dataset.sat_id.unique())

sats = []
point_names = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']

for sat_id in ids:

    predict = np.load('%s/%d.npy' % ('predicts', sat_id), allow_pickle=True).item()
    predict['sat_id'] = [sat_id] * len(predict['x'])
    predict['position'] = list(range(len(predict['x'])))

    sat_data = pd.DataFrame(predict)
    sat_data['sat_id_pos'] = sat_data['sat_id'].astype(str) + "_" + sat_data['position'].astype(str)
    sats.append(sat_data)

all_sats = pd.concat(sats)
del all_sats['position']
del all_sats['sat_id']
all_sats.to_csv('all_predictions.csv' ,index=False)


