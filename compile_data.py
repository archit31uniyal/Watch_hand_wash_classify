import pandas as pd
import numpy as np
import os
from collections import defaultdict

def generate_data(stride = 1, window_size = 1000):
    filename = f'features_stride_{1}_window_size_{window_size}.csv'

    # Read in the data
    wash_files = [fn for fn in os.listdir('./raw_data') if 'nonwash' not in fn]
    no_wash_files = [fn for fn in os.listdir('./raw_data') if 'nonwash' in fn]

    stats = defaultdict(list)
    for fn in wash_files:
        temp = pd.read_csv(os.path.join('./raw_data/', fn), names=['timestamp', 'x', 'y', 'z'])
        if len(temp)> window_size:
            # extra_entries = len(temp) % window_size
            # temp = temp[:-extra_entries]
            n = len(temp) - window_size   
            for i in range(1, n, stride):
                stats['mean_x'].append(np.mean(temp['x'][(i-1): i+window_size]))
                stats['std_x'].append(np.std(temp['x'][(i-1): i+window_size]))
                stats['mean_y'].append(np.mean(temp['y'][(i-1): i+window_size]))
                stats['std_y'].append(np.std(temp['y'][(i-1): i+window_size]))
                stats['mean_z'].append(np.mean(temp['z'][(i-1):  i+window_size]))
                stats['std_z'].append(np.std(temp['z'][(i-1): i+window_size]))
                stats['Activity'].append('hand_wash')


    for fn in no_wash_files:
        temp = pd.read_csv(os.path.join('./raw_data/',fn), names=['timestamp', 'x', 'y', 'z'])
        if len(temp) > window_size:
            n = len(temp) - window_size 
            for i in range(1, n):
                stats['mean_x'].append(np.mean(temp['x'][(i-1): i+window_size]))
                stats['std_x'].append(np.std(temp['x'][(i-1): i+window_size]))
                stats['mean_y'].append(np.mean(temp['y'][(i-1): i+window_size]))
                stats['std_y'].append(np.std(temp['y'][(i-1): i+window_size]))
                stats['mean_z'].append(np.mean(temp['z'][(i-1): i+window_size]))
                stats['std_z'].append(np.std(temp['z'][(i-1): i+window_size]))
                stats['Activity'].append('not_hand_wash')

    data = pd.DataFrame(stats)

    _, counts = list(np.unique(data['Activity'], return_counts=True))
    samples = min(counts)

    balanced_data = data.groupby("Activity").sample(n=samples, random_state=1)

    data.to_csv(filename, index=False)
    balanced_data.to_csv('balanced_' + filename, index=False)