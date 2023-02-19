from .model import Model
import torch
import os

class InstModel(Model):
    def __init__(self, model, model_name, device):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.history = {}

    def fit(self, epochs, train_loader, val_loader, criterion, optimizer, scheduler):
        pass
    
    def save(self, path_folder):
        torch.save(self.model, os.path.join(path_folder, self.model_name))
