import pandas as pd
import numpy as np
import os
from collections import defaultdict

def generate_data(window_size = 1000, add_extra_cols = False):
    filename = f'features_window_size_{window_size}_extra_cols_{add_extra_cols}_walk.csv'

    # Read in the data
    outdoor_walk_files = [fn for fn in os.listdir('./outdoor_walk') if '.csv' in fn]
    indoor_walk_files = [fn for fn in os.listdir('./indoor_walk') if '.csv' in fn]

    stats = defaultdict(list)
    for fn in outdoor_walk_files:
        temp = pd.read_csv(os.path.join('./outdoor_walk/', fn), names=['timestamp', 'x', 'y', 'z'])
        if len(temp) > window_size:
            # extra_entries = len(temp) % window_size
            # temp = temp[:-extra_entries]
            for i in range(0, len(temp), 1000):
                if i + window_size > len(temp):
                    break
                stats['mean_x'].append(np.mean(temp['x'][i: i+window_size]))
                stats['std_x'].append(np.std(temp['x'][i: i+window_size]))
                stats['mean_y'].append(np.mean(temp['y'][i: i+window_size]))
                stats['std_y'].append(np.std(temp['y'][i: i+window_size]))
                stats['mean_z'].append(np.mean(temp['z'][i: i+window_size]))
                stats['std_z'].append(np.std(temp['z'][i: i+window_size]))
                if add_extra_cols:
                    stats['median_x'].append(np.median(temp['x'][i: i+window_size]))
                    stats['median_y'].append(np.median(temp['y'][i: i+window_size]))
                    stats['median_z'].append(np.median(temp['z'][i: i+window_size]))
                    stats['rms_x'].append(np.sqrt(np.mean(temp['x'][i: i+window_size]**2)))
                    stats['rms_y'].append(np.sqrt(np.mean(temp['y'][i: i+window_size]**2)))
                    stats['rms_z'].append(np.sqrt(np.mean(temp['z'][i: i+window_size]**2)))
                stats['Activity'].append('outdoor walk')


    for fn in indoor_walk_files:
        temp = pd.read_csv(os.path.join('./indoor_walk/',fn), names=['timestamp', 'x', 'y', 'z'])
        if len(temp) > window_size:
            for i in range(0, len(temp), 1000):
                if i + window_size > len(temp):
                    break
                stats['mean_x'].append(np.mean(temp['x'][i: i+window_size]))
                stats['std_x'].append(np.std(temp['x'][i: i+window_size]))
                stats['mean_y'].append(np.mean(temp['y'][i: i+window_size]))
                stats['std_y'].append(np.std(temp['y'][i: i+window_size]))
                stats['mean_z'].append(np.mean(temp['z'][i: i+window_size]))
                stats['std_z'].append(np.std(temp['z'][i: i+window_size]))
                stats['Activity'].append('indoor walk')
                if add_extra_cols:
                    stats['median_x'].append(np.median(temp['x'][i: i+window_size]))
                    stats['median_y'].append(np.median(temp['y'][i: i+window_size]))
                    stats['median_z'].append(np.median(temp['z'][i: i+window_size]))
                    stats['rms_x'].append(np.sqrt(np.mean(temp['x'][i: i+window_size]**2)))
                    stats['rms_y'].append(np.sqrt(np.mean(temp['y'][i: i+window_size]**2)))
                    stats['rms_z'].append(np.sqrt(np.mean(temp['z'][i: i+window_size]**2)))

    data = pd.DataFrame(stats)

    _, counts = list(np.unique(data['Activity'], return_counts=True))
    samples = min(counts)

    balanced_data = data.groupby("Activity").sample(n=samples, random_state=1)

    data.to_csv(filename, index=False)
    balanced_data.to_csv('balanced_' + filename, index=False)

generate_data(window_size=4000, add_extra_cols=True)
print("Data generated successfully!")