import os
import numpy as np
import cv2
import random

def preprocessing(data):
    data /= 255.
    data -= 0.5
    data *= 2.
    return data


# original size 112x112, crop size 64x64
def load_data(image_path, flag):
    if flag == 'original':
        size = 112
    elif flag == 'crop':
        size = 64
    else:
        raise ('only support original and crop!')

    # total number of videos is 599
    # 16 person for training, random obtain 4 clips from every video, 1536 = 16x4x5x6
    # 9 person for validation, random obtain 1 clip from every video, 215 = 9x4x1x6-1
    seq_length = 16
    x_train = np.zeros((1536,seq_length,size,size,3),dtype='float32')
    y_train = []
    x_val = np.zeros((215,seq_length,size,size,3),dtype='float32')
    y_val = []

    actions = os.listdir(image_path)
    actions.sort(key=str.lower)
    train_num_count = 0
    total_count_train = 0
    total_count_val = 0
    label_count = 0
    for action in actions:
        print(action)
        samples = os.listdir(image_path+action)
        samples.sort(key=str.lower)
        for sample in samples:
            if train_num_count < 64:
                train_num_count += 1
                images = os.listdir(image_path+action+'/'+sample)
                images.sort(key=str.lower)
                frames = [i for i in range(len(images)-16)]
                start_frame = random.sample(frames,4)
                for frame in start_frame:
                    for j in range(seq_length):
                        img = cv2.imread(image_path+action+'/'+sample+'/'+images[frame+j])
                        img = cv2.resize(img,(size,size))
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        x_train[total_count_train,j,:,:,:] = img
                    total_count_train += 1
                    y_train.append(label_count)
            else:
                train_num_count += 1
                images = os.listdir(image_path + action + '/' + sample)
                images.sort(key=str.lower)
                frames = [i for i in range(len(images) - 16)]
                start_frame = random.sample(frames, 1)
                for frame in start_frame:
                    for j in range(seq_length):
                        img = cv2.imread(image_path + action + '/' + sample + '/' + images[frame+j])
                        img = cv2.resize(img, (size, size))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        x_val[total_count_val, j, :, :, :] = img
                    y_val.append(label_count)
                    total_count_val += 1
        label_count += 1
        train_num_count = 0
    x_train = preprocessing(x_train)
    x_train = np.transpose(x_train,(0,2,3,1,4))
    x_val = preprocessing(x_val)
    x_val = np.transpose(x_val,(0,2,3,1,4))
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    return x_train,y_train,x_val,y_val