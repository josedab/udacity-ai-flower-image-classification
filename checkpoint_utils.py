import torch
from model_utils import create_transfer_learning_model

categories_to_classify = 102

def save_model(file_path, architecture, model, hidden_layer_units):
    snapshot = {
        'architecture': architecture,
        'state': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_layer_units': hidden_layer_units
    }
    torch.save(snapshot, file_path)

def load_model(file_path):
    snapshot = torch.load(file_path, map_location=lambda storage, loc: storage)
    model = create_transfer_learning_model(snapshot['architecture'],
                                           snapshot['hidden_layer_units'],
                                           categories_to_classify)
    model.load_state_dict(snapshot['state'])
    model.class_to_idx = snapshot['class_to_idx']
    return model
