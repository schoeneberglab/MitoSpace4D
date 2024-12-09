import yaml

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

if __name__ == '__main__':
    cfg = load_config('/home/dhruvagarwal/projects/Manav_MitoSpace/MitoSpace4D/autoencoder/config.yaml')
    print(cfg)