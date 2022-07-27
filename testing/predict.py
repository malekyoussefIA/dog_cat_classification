import torch
from training.data import DataloaderDogCat
from training.model import get_model
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np
import yaml
from training.utils import accuracy,clean_folder


def predict(prediction_config='./configs/predict.yaml'):
    with open(prediction_config, 'r') as f:
        configuration = OmegaConf.create(yaml.safe_load(f))
    clean_folder(configuration.data.path)
    test_data =  DataloaderDogCat(configuration.data.path,kind = 'test')
    test_loader = DataLoader(test_data, configuration.batch_size, shuffle=True)
    checkpoint_path = configuration.checkpoint.path

   #get model and load the checkpoint
    model = get_model(configuration.model_to_load)
    ckpt = torch.load(checkpoint_path) 
    model.load_state_dict(ckpt,strict=True)

    #check if GPUs are available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.freeze()
    model.to(device)

    test_accuracy=[]

    #predict
    for images, labels in test_loader:
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))
        preds = model(images)
        print(type(preds))
        print(type(images))

        #Calculating Accuracy
        acc = accuracy(preds, labels)
        test_accuracy.append(acc)
    print(f"Prediction accuraccy --> {np.mean(test_accuracy)}")
        #Calculating precision and recall
         #(to do)

if __name__ =='__main__': 
    predict()