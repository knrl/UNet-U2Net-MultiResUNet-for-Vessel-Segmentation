import os
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow

def save_pickle(data, path):
    import pickle
    pickle.dump(data, open(path, 'wb'))

def transform_data(image, mask, IMG_HEIGHT, IMG_WIDTH):
    transformed_image = np.zeros((6, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    transformed_mask = np.zeros((6, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    transform1 = A.Compose([A.RandomCrop(width=IMG_WIDTH-384, height=IMG_HEIGHT-384)])
    transform2 = A.Compose([A.HorizontalFlip(p=1)])
    transform3 = A.Compose([A.CLAHE()])
    transform4 = A.Compose([A.RandomRotate90(p=1)])
    transform5 = A.Compose([A.VerticalFlip(p=1)])

    transformed_image[0] = np.expand_dims(resize(image, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
    transformed_mask[0] = np.expand_dims(resize(mask, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
    for i in range(5):
        if (i == 0):
            transformed = transform1(image=image, mask=mask)
        elif (i == 1):
            transformed = transform2(image=image, mask=mask)
        elif (i == 2):
            transformed = transform3(image=image, mask=mask)
        elif (i == 3):
            transformed = transform4(image=image, mask=mask)
        elif (i == 4):
            transformed = transform5(image=image, mask=mask)
            
        transformed_image[i+1] = np.expand_dims(resize(transformed['image'], (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
        transformed_mask[i+1] = np.expand_dims(resize(transformed['mask'], (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
    return transformed_image, transformed_mask


def train_and_mask_dataset(TRAIN_PATH, MASK_TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, TARGET_H, TARGET_W):
    t1_list, t2_list = os.listdir(TRAIN_PATH), os.listdir(MASK_TRAIN_PATH)
    t1_list.sort()
    t2_list.sort()
    t1_list, t2_list = t1_list[:30], t2_list[:30]
    
    n,j = 0,0
    num_of_filters = 6
    # Eger .Ds_store formatında dosya varsa number_of_file her .ds_ dosyası için -1 azaltılmalıdır.
    num_of_image = len(t1_list) - 1
    h = int(IMG_HEIGHT/TARGET_H)
    w = int(IMG_WIDTH/TARGET_W)
    X_train = np.zeros((h*w*num_of_filters*num_of_image,TARGET_H,TARGET_W,1), dtype=np.uint8)
    Y_train = np.zeros((h*w*num_of_filters*num_of_image,TARGET_H,TARGET_W,1), dtype=np.bool)

    for (id_, id_m) in zip(t1_list, t2_list):
        if (id_[0] == '.' or id_m[0] == '.'):
            continue
        path = TRAIN_PATH + id_
        mask_path = MASK_TRAIN_PATH + id_m
        
        img = imread(path, 0)[:,:,1]
        mask_img = imread(mask_path, 0)

        transformed_data = transform_data(img, mask_img, IMG_HEIGHT, IMG_WIDTH)
        for i in range(num_of_filters):
            for i1 in range(h):
                for i2 in range(w):
                    X_train[n] = np.array(transformed_data[0][i][i1*TARGET_H:i1*TARGET_H+TARGET_H, i2*TARGET_W:i2*TARGET_W+TARGET_W])
                    Y_train[n] = np.array(transformed_data[1][i][i1*TARGET_H:i1*TARGET_H+TARGET_H, i2*TARGET_W:i2*TARGET_W+TARGET_W])
                    n += 1
        j += 1

    print("Train images and masks done.", X_train.shape, " ", Y_train.shape)
    
    return X_train, Y_train


def test_and_mask_dataset(TEST_PATH, MASK_TEST_PATH, IMG_HEIGHT, IMG_WIDTH, TARGET_H, TARGET_W):
    t1_list, t2_list = os.listdir(TEST_PATH), os.listdir(MASK_TEST_PATH)
    t1_list.sort()
    t2_list.sort()
    t1_list, t2_list = t1_list[30:], t2_list[30:]
    
    n = 0
    num_of_image = len(t1_list)
    h = int(IMG_HEIGHT/TARGET_H)
    w = int(IMG_WIDTH/TARGET_W)
    X_test = np.zeros((h*w*num_of_image,TARGET_H,TARGET_W,1), dtype=np.uint8)
    Y_test = np.zeros((h*w*num_of_image,TARGET_H,TARGET_W,1), dtype=np.uint8)
    
    for (id_, id_m) in zip(t1_list, t2_list):
        if (id_[0] == '.' or id_m[0] == '.'):
            continue
        test_path = TEST_PATH + id_
        test_img = imread(test_path, 0)[:,:,1]
        test_img = np.expand_dims(resize(test_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
        
        mask_test_path = MASK_TEST_PATH + id_m
        mask_test_img = imread(mask_test_path, 0)
        mask_test_img = np.expand_dims(resize(mask_test_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),axis=-1)
        for i1 in range(h):
            for i2 in range(w):
                X_test[n] = test_img[i1*TARGET_H:i1*TARGET_H+TARGET_H, i2*TARGET_W:i2*TARGET_W+TARGET_W]
                Y_test[n] = mask_test_img[i1*TARGET_H:i1*TARGET_H+TARGET_H, i2*TARGET_W:i2*TARGET_W+TARGET_W]
                n += 1

    print("Test images done. X_tes: ", X_test.shape, " Y_test: ", Y_test.shape)
    
    return X_test, Y_test

TRAIN_PATH = ''
MASK_TRAIN_PATH = ''
MASK_PATH = ''

IMG_WIDTH = 3072
IMG_HEIGHT = 2048
IMG_CHANNELS = 3
TARGET_H = 128
TARGET_W = 128


X_train, Y_train = train_and_mask_dataset(TRAIN_PATH, MASK_TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, TARGET_H, TARGET_W)
X_test, Y_test = test_and_mask_dataset(TRAIN_PATH, MASK_TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, TARGET_H, TARGET_W)
Test_mask, _ = test_and_mask_dataset(MASK_PATH, MASK_TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, TARGET_H, TARGET_W)

save_pickle(X_train, 'X_train.pickle')
save_pickle(Y_train, 'Y_train.pickle')
save_pickle(X_test, 'X_test.pickle')
save_pickle(Y_test, 'Y_test.pickle')
save_pickle(Test_mask, 'X_test_mask.pickle')

#j = number_of_path * number_of_retina_image
j = 0
full_img = np.zeros((IMG_HEIGHT, IMG_WIDTH))
for i in range(IMG_HEIGHT/TARGET_H):
    for k in range(IMG_WIDTH/TARGET_W):
        full_img[i*TARGET_H:i*TARGET_H+TARGET_H, k*TARGET_W:k*TARGET_W+TARGET_W] = X_train[j].reshape(TARGET_H,TARGET_W)
        j += 1

plt.imshow(full_img, cmap='gray')