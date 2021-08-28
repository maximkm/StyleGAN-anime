from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import sys
from PIL import Image
from tqdm import tqdm
import importlib
import wandb
import json
import os


def dynamic_import(module):
    return importlib.import_module(module)


def get_py_modules(files_dir):
    files = []
    for file in os.listdir(files_dir):
        if '.py' in file:
            files.append(file[:-3])
    return files


class ImageDataset(Dataset):
    def __init__(self, data_root, transform):
        self.samples = []
        self.transform = transform

        for class_dir in os.listdir(data_root):
            data_folder = os.path.join(data_root, class_dir)

            for image_dir in tqdm(os.listdir(data_folder)):
                img = Image.open(f'{data_folder}/{image_dir}')
                img = img.convert("RGB")
                self.samples.append(self.transform(img))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def init_train(path_to_config, wandb_set=False, load_dataset=True):
    # Loading configurations
    with open(path_to_config, "r") as fp:
        conf = json.load(fp)
    conf['wandb_set'] = wandb_set    
    
    # Loading all models
    generators = {}
    discriminators = {}
    for name_model in get_py_modules('models'):
        model = dynamic_import(f'models.{name_model}')
        generators = {**generators, **model.generators}
        discriminators = {**discriminators, **model.discriminators}
    assert conf["Generator"] in generators.keys()
    assert conf["Discriminator"] in discriminators.keys()
    
    # Loading the loss functions
    losses = dynamic_import('losses')
    losses_gen = losses.gen_losses
    losses_disc = losses.disc_losses
    assert conf["Loss_gen"] in losses_gen.keys()
    assert conf["Loss_disc"] in losses_disc.keys()
    
    # Loading the trainer
    trainers = dynamic_import('trainer').trainers
    assert conf["Trainer"] in trainers.keys()
    
    # Checking the optimizer
    assert hasattr(torch.optim, conf['Optim_G'])
    assert hasattr(torch.optim, conf['Optim_D'])
        
    # Init models
    start_epoch = 0
    G = generators[conf['Generator']](**conf['Gen_config'])
    D = discriminators[conf['Discriminator']](**conf['Disc_config'])
    print(G)
    print(D)

    # Load the pre-trained weight
    conf["Weight_dir"] = os.path.join(conf["Weight_dir"], f'{conf["Generator"]} {conf["Discriminator"]} {conf["IMG_SIZE"]}')
    if os.path.exists(conf["Weight_dir"]):
        name_to_epoch = lambda x: int(x.replace('.pth', '').replace('weight ', ''))
        epochs = sorted([name_to_epoch(elem) for elem in os.listdir(conf["Weight_dir"]) if '.pth' in elem])
        if len(epochs) > 0:
            last_epoch = epochs[-1]
            print(f'{conf["Weight_dir"]}/weight {last_epoch}.pth')
            state = torch.load(f'{conf["Weight_dir"]}/weight {last_epoch}.pth')
            print(f'Load the pre-trained weight {last_epoch}')
            G.load_state_dict(state['G'])
            D.load_state_dict(state['D'])
            start_epoch = state['start_epoch']
    else:
        os.makedirs(conf["Weight_dir"])

    # Multi-GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f'Avalible {torch.cuda.device_count()} GPUs')
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    G.to(device)
    D.to(device)

    # Create the criterion, optimizer
    optim_G = getattr(torch.optim, conf['Optim_G'])(G.parameters(), **conf["Optim_G_config"])
    optim_D = getattr(torch.optim, conf['Optim_D'])(D.parameters(), **conf["Optim_D_config"])
    
    # Load train image
    transform = transforms.Compose([
        transforms.Resize((conf["IMG_SIZE"], conf["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
    ])

    if load_dataset:
        dataset = ImageDataset(conf["Dataset"], transform=transform)
        dataloader = DataLoader(dataset, batch_size=conf["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    else:
        dataloader = None
    
    # Packaging of parameters for the trainer
    form_for_trainer = {
        "G": G,
        "D": D,
        "start_epoch": start_epoch,
        "dataloader": dataloader,
        "optim_G": optim_G,
        "optim_D": optim_D,
        "gen_loss": losses_gen[conf["Loss_gen"]],
        "disc_loss":  losses_disc[conf["Loss_disc"]],
        "z_dim": conf["z_dim"],
        "device": device,
    }
    
    Trainer = trainers[conf["Trainer"]](conf, **form_for_trainer)
    return Trainer


if __name__ == "__main__":
    assert len(sys.argv) <= 3
    wandb_set = False if len(sys.argv) == 2 else (sys.argv[-1] == "True")
    
    Trainer = init_train(sys.argv[1], wandb_set)
    Trainer.train_loop()
