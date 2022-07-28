import numpy as np 
import os
import sys
import cv2
from sklearn.metrics import precision_score, recall_score
import  matplotlib.pyplot  as plt
from albumentations.augmentations import Normalize


def denormalize(image):
    #inverse albumentations normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    denorm = Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        always_apply=True,
        max_pixel_value=1.0
    )
    image= image.cpu().numpy()
    image = np.transpose(image,(1,2,0))
    image = (denorm(image=image)["image"]*255).astype(np.uint8)
    return image


def accuracy(preds, trues):
    #calculate prediction accuracy 
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)


def precision_recall(preds,trues):
    #calculate precision and recall
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    trues = [int(trues[i].item()) for i in range(len(preds))]
    prec = precision_score(preds,trues)
    rec = recall_score(preds,trues)
    return prec,rec
    


def clean_folder(folder_path):
    # set an initial value which no image will meet
    minw,minh = 128,128
    filenames = [f for f in os.listdir(folder_path) if not f.startswith('.')]   #to avoid hidden files
    for image_name in filenames:
        try :
            full_path = os.path.join(folder_path, image_name)
            img = cv2.imread(full_path)
            h,w = img.shape[0],img.shape[1]
            if h < minh or w < minw:
                os.remove(full_path)
        except AttributeError as e:
            print(f"Something went wrong with {image_name}. Image was removed...")
            os.remove(full_path)


def save_images(images,preds, trues,bad_images_folder):
    #convert pytorch tensors and save them as images to folder
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    trues = [int(trues[i].item()) for i in range(len(preds))]
    for j in range(len(preds)):
        if preds[j]!=trues[j]:
            image = denormalize(images[j])
            cv2.imwrite(bad_images_folder+"test"+str(j)+".jpg",image)


            

class EarlyStopping():
    #early stopping to avoid overfitting
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