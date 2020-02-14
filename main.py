import numpy as np
import pandas as pd

test_dataset = pd.read_csv('test.csv', usecols=['id', 'epoch', 'sat_id'], parse_dates=['epoch'])
test_dataset['millis'] = test_dataset.epoch.astype(np.int64)
test_dataset['diff'] = test_dataset.groupby('sat_id').millis.shift(-1) - test_dataset.millis
test_dataset = test_dataset.sort_values(['sat_id', 'millis'])
test_dataset_duplicates = test_dataset[test_dataset['diff'] < 1.0e+9]
test_dataset = test_dataset[~(test_dataset['diff'] < 1.0e+9)]

test_dataset = test_dataset.assign(position=test_dataset.groupby(['sat_id']).cumcount())
test_dataset['sat_id_pos'] = test_dataset['sat_id'].astype(str) + "_" + test_dataset['position'].astype(str)
test_dataset = test_dataset.set_index('sat_id_pos')
all_predictions = pd.read_csv('all_predictions.csv', index_col='sat_id_pos')
labeled_dataset = test_dataset.join(all_predictions, how='left')
duplicates = labeled_dataset[labeled_dataset.id.isin(test_dataset_duplicates.id.values + 1)]
duplicates['id'] = test_dataset_duplicates.id.values
all_data = pd.concat([labeled_dataset, duplicates])
all_data.sort_values('id')[['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']].to_csv('submission.csv', index=False)
