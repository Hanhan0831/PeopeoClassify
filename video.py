import numpy as np
import pandas as pd
from tqdm.auto import tqdm
data_with_labels=pd.read_csv('Experiment_1/Experiment_1.csv')
#新建一个空的dataframe
df = pd.DataFrame(columns=['UnitQuaternion.x', 'UnitQuaternion.y', 'UnitQuaternion.z', 'UnitQuaternion.w', 'HmdPosition.x', 'HmdPosition.y', 'HmdPosition.z','objectx', 'objecty'])
for i in tqdm(range(0,9)):
    # 只读取data_with_labels中video_id为1的数据
    data_chunk = data_with_labels[data_with_labels['video_id'] == i]
    filepath = f'Experiment_1/video_{i}.csv'
    video = pd.read_csv(filepath)[['time', 'x', 'y']]
    def search_video(search_value):

        search_value_str = f"{search_value:.2f}"
        result = video.loc[video['time'].astype(str).str[:4] == search_value_str[:4]].head(1)
        if len(result) > 0:
            return np.float64(result['x']),np.float64(result['y'])
        else:
            return float(0),float(0)
    i = 0

    for ind, row in tqdm(data_chunk.iterrows(),total=len(data_chunk)):
        time=row['PlaybackTime']
        x,y=search_video(time)
        row['objectx']=np.float64(row['Locationx'])-(x)
        row['objecty']=np.float64(row['Locationy'])-(y)
        i += 1
        df = df.append(row, ignore_index=True)