from train import train
from omegaconf import OmegaConf
import yaml




def main(training_config='./configs/default.yaml'):
    with open(training_config, 'r') as f:
        configuration = OmegaConf.create(yaml.safe_load(f))

    train(configuration.training.epochs,
          configuration.optimizer.lr,
          configuration.training.batch_size,
          configuration.training.model_to_load,
          configuration.training.dataset_path,
          configuration.training.ckpt_path)


if __name__ =='__main__': 
    main()