from pathlib import Path

import numpy as np
import torch
import cv2
import time
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO
import pandas as pd

def abstract_video(filename):
#保存csv文件
    print(filename)
    df = pd.DataFrame(columns=['class_id', 'x', 'y','time'])

    cap = cv2.VideoCapture(filename=filename)
    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 保存每一帧的时间戳

    timestamps = np.zeros(total_frames)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #获取当前帧时间戳
        timestamps[frame_count] = cap.get(cv2.CAP_PROP_POS_MSEC)
        #转换时间戳为秒
        timestamps[frame_count] /= 1000.0
        results = model(frame)
        boxes = results[0].boxes
        #判断是否检测到物体
        if boxes.shape[0] > 0:
            box = boxes[0]  # returns one
            if box.cls == 0:
                print("People")
                for obj in box.xywh:
                    x, y = float(obj[0]), float(obj[1])
                    df = df.append({'class_id': "People", 'x': x, 'y': y,'time':timestamps[frame_count]}, ignore_index=True)
                    print( x, y)

            # 更新帧计数器
        frame_count += 1

        # 跳过接下来的 (1/fps - 0.1) 秒的边界
        if frame_count % int(fps / 10) == 0:
            # 输出当前进度
            print(f'Processed {frame_count}/{total_frames} frames')
    df.to_csv(filename, index=False, encoding='utf_8_sig')
model = YOLO('yolov8n.pt')
#读取文件夹内所有文件
path = Path('Experiment_1')
files = list(path.glob('*.mp4'))
for file in tqdm(files, desc='Processing videos'):
    abstract_video(str(file))

