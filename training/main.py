from train import train
from omegaconf import OmegaConf
import yaml




def main(training_config='./configs/default.yaml'):
    with open(training_config, 'r') as f:
        configuration = OmegaConf.create(yaml.safe_load(f))

    train(configuration.training.epochs,
          configuration.optimizer.lr,
          configuration.training.model_to_load,
          configuration.dataset_path,
          configuration.ckpt_path)


if __name__ =='__main__': 
    main()