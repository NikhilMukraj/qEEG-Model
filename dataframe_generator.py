import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os



path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
filename = [i for i in os.listdir() if '.csv' in i][0]
df = pd.read_csv(f'{path}\\{filename}')

nans = []
for i in df.iloc[:, 8:]:
    nans.append(df[i].isnull().to_list())

nan_stats = [len([i for i in nans[j] if i]) for j in range(len(nans))]

has_nans = [i for i in nan_stats if i > 0]
to_drop = []
for i in has_nans:
    to_drop.append(nan_stats.index(i))

df = df.drop([df.columns[i+8] for i in to_drop], axis=1)

xdata = df.iloc[:, 8:]
ydata = df['main.disorder']

xdata.to_csv(f'{path}\\data\\raw_xdata.csv', index=False)
ydata.to_csv(f'{path}\\data\\raw_ydata.csv', index=False)

one_hot_y_data =  pd.get_dummies(df['main.disorder'].to_list())
one_hot_y_data.to_csv(f'{path}\\data\\ydata.csv', index=False)

scale = np.max(np.array(df.iloc[:, 8:]))
data = np.array(df.iloc[:, 8:]) / scale

pca = PCA(n_components=81)
transformed = pca.fit_transform(data)

pca_data = pd.DataFrame(transformed)
pca_data.to_csv(f'{path}\\data\\pca_data.csv', index=False)