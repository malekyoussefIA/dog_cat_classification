import numpy as np 
import os
import sys
import cv2


def accuracy(preds, trues):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)


def clean_folder(folder_path):
    # set an initial value which no image will meet
    minw = 128
    minh = 128
    filenames = os.listdir(folder_path)
    for image_name in filenames:
        img = cv2.imread(folder_path+image_name)
        h,w = img.shape[0],img.shape[1]     
        if h < minh or w < minw:
                os.remove(folder_path+image_name)
            

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True