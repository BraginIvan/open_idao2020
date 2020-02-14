1. run generate_predicts.py
- It finds parameters (a,V,scale,add) of function sin(a\*t+V)\*scale\*t+add\*t (instead of sin it uses a period of satellite) for each value (x,y,z,Vx,Vy,Vz).
- It finds parameters to minimize MAE loss
- Then it applies the function for next month values. We do not use simulated data but only time so we can predict values for both tracks offline.
2. run npy_to_pandas.py
- It reads all predictions for all satellites from npy 
- Creates pandas dataframe
- Saves the dataframe as csv
3. run main.py
- It reads test.csv
- Removes duplicates,
- Reads predicted values from all_predictions.csv
- Joins test with predicted values 
- Moves duplicates back. 