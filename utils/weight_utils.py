import torch

def get_weights_copy(model_state):
    weights_path = 'weights_temp.pt'
    torch.save(model_state, weights_path)
    return torch.load(weights_path)