from cgi import test
from tqdm import tqdm
from unittest import TestCase, TextTestResult
from data import DataloaderDogCat,get_data
import numpy as np
import torch.nn as nn 
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from model import get_model
from utils import accuracy,clean_folder,EarlyStopping
import wandb
from efficientnet_pytorch import EfficientNet

wandb.init(project='dog_cat_classification')

def do_epoch(model,train_loader,val_loader,device,optimizer,criterion,ckpt_path):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
   #### Training iteration ###
    for images, labels in train_loader:
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))
        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(images)
        #Calculating Loss
        loss = criterion(preds, labels)
        # Backward
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss.append(loss.item())
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        train_accuracy.append(acc)

    #### validation iteration ###
    min_loss = 1
    for images, labels in val_loader:
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))
        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(images)
        #Calculating Loss
        loss = criterion(preds, labels)
        # Calculate Loss
        val_loss.append(loss.item())
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        val_accuracy.append(acc)

# Saving best checkpoints
    if min_loss > np.mean(val_loss):
        min_loss = np.mean(val_loss)
    torch.save(model.state_dict(), ckpt_path+'best.ckpt')
        
    return np.mean(train_loss),np.mean(train_accuracy),np.mean(val_loss),np.mean(val_accuracy)




 
def train(epochs,lr,batch_size,model_to_load,dataset_path,ckpt_path):
    #clean data, as some images are two small (50*59) so all images under 128*128 will be removed
    clean_folder(dataset_path+'train/')
    clean_folder(dataset_path+'val/')
    #check if GPUs are available(to make training faster)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #initialise dataloader
    train_data,val_data = get_data(dataset_path)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)
    


    #load model --> "Resnet50"  or  'efficientnet-b0'
    model = get_model(model_to_load)
              
    # define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    model.to(device)

    early_stopping = EarlyStopping(tolerance=4, min_delta=0.1)  


    for epoch in tqdm(range(epochs)):
        #train
        train_loss, train_accuracy,val_loss,val_accuracy  = do_epoch(model,train_loader,val_loader,device,optimizer,criterion,ckpt_path)
        wandb.log({'train_loss':train_loss,'custom_step': epoch})
        wandb.log({'train_accuracy':train_accuracy,'custom_step': epoch})
        wandb.log({'val_loss':val_loss,'custom_step': epoch})
        wandb.log({'val_accuracy':val_accuracy,'custom_step': epoch})

        # early stopping to avoid overfitting
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)

        print(f"train_loss : {train_loss}, train_accuracy : {train_accuracy}")
        print(f"val_loss : {val_loss}, val_accuracy : {val_accuracy}")
        
        