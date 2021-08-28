import torch
import os


def LoadWeights(Trainer, file):
    assert os.path.exists(file)
    state = torch.load(file)
    fix_state = {'G': {}, 'D': {}}
    for key, value in state['G'].items():
        fix_state['G'][key.replace('module.', '')] = value
    for key, value in state['D'].items():
        fix_state['D'][key.replace('module.', '')] = value
    Trainer.G.load_state_dict(fix_state['G'], strict=False)
    Trainer.D.load_state_dict(fix_state['D'], strict=False)
