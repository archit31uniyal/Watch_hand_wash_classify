import pandas as pd
import numpy as np
import os
from collections import defaultdict

filename = 'features.csv'

# Read in the data
wash_files = [fn for fn in os.listdir('./raw_data') if 'nonwash' not in fn]
no_wash_files = [fn for fn in os.listdir('./raw_data') if 'nonwash' in fn]

stats = defaultdict(list)
for fn in wash_files:
    temp = pd.read_csv(os.path.join('./raw_data/', fn), names=['timestamp', 'x', 'y', 'z'])
    if len(temp)> 1000:
        extra_entries = len(temp) % 1000
        temp = temp[:-extra_entries]
        n = int(len(temp)/1000)    
        for i in range(1, n):
            stats['mean_x'].append(np.mean(temp['x'][(i-1)*1000:i*1000]))
            stats['std_x'].append(np.std(temp['x'][(i-1)*1000:i*1000]))
            stats['mean_y'].append(np.mean(temp['y'][(i-1)*1000:i*1000]))
            stats['std_y'].append(np.std(temp['y'][(i-1)*1000:i*1000]))
            stats['mean_z'].append(np.mean(temp['z'][(i-1)*1000:i*1000]))
            stats['std_z'].append(np.std(temp['z'][(i-1)*1000:i*1000]))
            stats['Activity'].append('hand_wash')

print(no_wash_files)
for fn in no_wash_files:
    temp = pd.read_csv(os.path.join('./raw_data/',fn), names=['timestamp', 'x', 'y', 'z'])
    if len(temp) > 1000:
        extra_entries = len(temp) % 1000
        temp = temp[:-extra_entries]
        n = int(len(temp)/1000)
        for i in range(1, n):
            stats['mean_x'].append(np.mean(temp['x'][(i-1)*1000:i*1000]))
            stats['std_x'].append(np.std(temp['x'][(i-1)*1000:i*1000]))
            stats['mean_y'].append(np.mean(temp['y'][(i-1)*1000:i*1000]))
            stats['std_y'].append(np.std(temp['y'][(i-1)*1000:i*1000]))
            stats['mean_z'].append(np.mean(temp['z'][(i-1)*1000:i*1000]))
            stats['std_z'].append(np.std(temp['z'][(i-1)*1000:i*1000]))
            stats['Activity'].append('not_hand_wash')

print(stats['mean_x'])
data = pd.DataFrame(stats)

data.to_csv(filename, index=False)