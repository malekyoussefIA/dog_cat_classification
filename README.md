# Dogs and Cats classification
​
This project aims at classifying images that depict either dogs or cats. 
​
## Table of contents
* [General information](#general-info)
	* [Dataset](#installation)
	* [Prerequies](#prerequies)
* [Installation](#installation)
* [Usage](#usage)
	* [Training](#training)
	* [Inference](#inference)
* [To improve](#to-improve)
​
## General information
The code allow us to train and test a deep neural network that is extensively devoted to classify dogs and cats images. Trainings logs are saved on a dedicated Weights&Biases project page available [here](https://wandb.ai/malekyoussef/dog_cat_classification?workspace=user-malekyoussef). 
​
### Dataset
The dataset used for training is publicly available [here](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). 
​
From this full set of images, 18 has been discarded from the training dataset and 11 from the validation one, since their size were below the minimal acceptable size (128x128). 
​
From a general perspective: 
	- The **.jpeg** images used for training were resized to 128x128. 
	
	-  The dataset contains N images of *dogs* and M of *cats*, total 2000 images with their corresponding labels.
	
	- A train / val / test slip has been made and respectively set to 80% / 10% / 10%. 
	
	- Some basic data-augmentation techniques were used during training (random horizontal flipping, 90° rotation... etc).
​
### Prerequies
​
Such code can be smoothly run on any Unix system with Python installed. Even though a GPU is highly encouraged to speed up both training and inference time processing, the model can be used as-is on CPU. 
​
The whole code is build opon the PyTorch deep learning framework, and has only few requirements regarding its esssential library. 
​
​
## Installation

##Installation
Clone the repo first : 
```
git clone git@github.com:malekyoussefIA/dog_cat_classification.git
```
and 
```
cd /dog_cat_classification 
```
​
* To build up a new virtual environment:
```python 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
​
## Usage 
### Training
Please make sure to update the *.yaml* file in `configs/default.yaml` where all meaningful hyperparameters can be changed by user. 
​
Then, training can be launch (either on GPU or CPU) with : 
```bash python -m training.main```
​
### Inference
First thing to do is to make sure to update the *.yaml* file in `configs/predict.yaml`. 
Then, inference can be performed through the command line: 
```bash python -m testing.predict```
​
​
## To do 
Several features might be added or refined in this project: 
​
* Made the training multi-GPU distributed. 
* Improve the data augmentation strategy that was implemented. 
