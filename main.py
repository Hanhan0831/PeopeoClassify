#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

# 初始化一个空的数据列表，用于存储数据和标签
data_list = []

# 遍历Experiment_1目录下的48个文件夹
for user_id in range(1, 49):
    user_folder = os.path.join("Experiment_1", str(user_id))

    # 遍历每个用户的8个视频文件
    for video_id in range(1, 9):
        video_file = os.path.join(user_folder, f"video_{video_id}.csv")

        # 读取CSV文件，跳过第一行标题
        df = pd.read_csv(video_file, skiprows=1, header=None)

        # 为DataFrame添加列名
        df.columns = ["Timestamp", "PlaybackTime", "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z",
                      "UnitQuaternion.w", "HmdPosition.x", "HmdPosition.y", "HmdPosition.z"]

        # 添加标签列，表示用户ID
        df["User_ID"] = user_id

        # 将数据添加到数据列表中
        data_list.append(df)

# 合并所有数据到一个大的DataFrame
all_data = pd.concat(data_list, ignore_index=True)

# 转换为NumPy数组，以便进一步处理
data_array = all_data.to_numpy()

print(data_array)


# In[23]:


from tqdm.auto import tqdm
import pandas as pd
import numpy as np

def autocorr(x, max_lag):
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    result /= result[0]
    return result[:max_lag + 1]

max_lag = 5
sample_interval = 0.1
num_features = 7
samples_per_sec = 10

# 初始化一个空的数据列表，用于存储特征和标签
data_list = []

# 遍历每个用户
for user_id in tqdm(range(1, 49)):
    # 遍历每个用户的每个视频
    for video_id in range(1, 9):
        df = data_array[data_array[:, -1] == user_id]
        df = df[df[:, 1] >= (video_id - 1) * 100]
        df = df[df[:, 1] < video_id * 100]

        if df.shape[0] == 0:
            continue

        # 对每一秒的数据进行处理
        for sec in range(int(df[:, 1].min()), int(df[:, 1].max()) + 1):
            df_sec = df[(df[:, 1] >= sec) & (df[:, 1] < sec + 1)]

            # 对每秒的数据进行等间隔采样
            df_resampled = pd.DataFrame(index=np.arange(0, 1, sample_interval))
            for col in range(2, 2 + num_features):
                df_resampled[str(col)] = np.interp(df_resampled.index, df_sec[:, 1].astype(float) - sec, df_sec[:, col].astype(float))

            # 计算自相关特征
            autocorr_features = []
            for col in range(2, 2 + num_features):
                autocorr_features.extend(autocorr(df_resampled[str(col)].values, max_lag))

            # 添加这一秒的特征和标签到数据列表
            for _ in range(samples_per_sec):
                data_list.append([user_id] + autocorr_features)

# 将数据列表转换为NumPy数组
data_array_processed = np.array(data_list)

print("Data array shape:", data_array_processed.shape)


# In[24]:


print(data_array_processed)
print(len(data_array_processed))


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 准备数据
X = data_array_processed[:, 1:]
y = data_array_processed[:, 0]

# 数据归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 将标签减1，使其范围为0到47
y = y - 1

# 对标签进行one-hot编码
y = np.eye(48)[y.astype(int)]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 为输入数据增加一个维度
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 创建双向LSTM模型
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(48, activation='softmax'))

model.summary()

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)


# In[ ]:




