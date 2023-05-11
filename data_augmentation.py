import pandas as pd
import numpy as np
import sys
import os


path = os.path.dirname(os.path.abspath(__file__)) 
pcapath = f'{path}\\data\\pca_data.csv'
ypath = f'{path}\\data\\ydata.csv'

data = pd.read_csv(pcapath)
ydata = pd.read_csv(ypath)

data = np.array(data)
ydata = np.array(ydata)

augmented = []
y_augmented = []

for n, i in enumerate(data):
    for _ in range(10):
        augmented.append(i + (np.random.randn(*i.shape) / 17.5))
        y_augmented.append(ydata[n])

augmented = np.array(augmented)
y_augmented = np.array(y_augmented)

pd.DataFrame(augmented).to_csv(f'{path}\\data\\augmented_x.csv', index=False)
pd.DataFrame(y_augmented).to_csv(f'{path}\\data\\augmented_y.csv', index=False)