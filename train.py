from data import get_dataset


# prepare data
def get_data():
    train_data = get_dataset('./dog_cat_classification/train')
    val_data = get_dataset('./dog_cat_classification/val')



if __name__ == '__main__':
    get_data()