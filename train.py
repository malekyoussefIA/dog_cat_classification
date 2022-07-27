from cgi import test
from tqdm import tqdm
from unittest import TestCase, TextTestResult
from data import DataloaderDogCat
import numpy as np
import torch.nn as nn 
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader,get_data
from torchvision import models
from utils import accuracy,clean_folder,EarlyStopping
import wandb
wandb.init(project='dog_cat_classification')

def do_epoch(model,loader,device,optimizer,criterion):
    epoch_loss = []
    epoch_accuracy = []
    
   # iteration 
    for images, labels in loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)

        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_accuracy.append(acc)


        return np.mean(epoch_loss),np.mean(epoch_accuracy)


def do_epoch_test(model,loader,device,optimizer,criterion):
    epoch_loss = []
    epoch_accuracy = []
    
   # iteration 
    for images, labels in loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape    
        #Calculating Loss
        preds = model(images)
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_accuracy.append(acc)

        return np.mean(epoch_loss),np.mean(epoch_accuracy)

 
def main():
    #clean data, as some images are two small (50*59) so all images under 128*128 will be removed
    clean_folder('train/')
    clean_folder('val/')
    clean_folder('test/')
    #check if GPUs are available(to make training faster)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #initialise dataloader
    train_data,val_data,test_data = get_data()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


    #load model --> "Resnet50"  or  "DenseNet121"

                     
    # define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)




    epochs = 32
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)
    #train our model
    for epoch in tqdm(range(epochs)):
        #train
        train_loss, train_accuracy  = do_epoch(model,train_loader,device,optimizer,criterion)
        wandb.log({'train_loss':train_loss,'custom_step': epoch})
        wandb.log({'train_accuracy':train_accuracy,'custom_step': epoch})
        #validate
        val_loss, val_accuracy  = do_epoch(model,val_loader,device,optimizer,criterion)
        wandb.log({'val_loss':val_loss,'custom_step': epoch})
        wandb.log({'val_accuracy':val_accuracy,'custom_step': epoch})

        # early stopping to avoid overfitting
        early_stopping(train_loss, val_loss)
        #if early_stopping.early_stop:
        #    print("We are at epoch:", epoch)
        #    break


        #test
        test_loss, test_accuracy  = do_epoch_test(model,test_loader,device,optimizer,criterion)

        print(f"train_loss : {train_loss}, train_accuracy : {train_accuracy}")
        print(f"val_loss : {val_loss}, val_accuracy : {val_accuracy}")
        
        






if __name__ == '__main__':
    main()