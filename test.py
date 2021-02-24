import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

model = PretrainedResNet(512)
model.load_state_dict(torch.load("../arcface/logs/arcface_SGD_epochs200_batch_size64_lr0.01_lrdecay0.00E+00_stepsize20_margin28.6_scale64/arcface_SGD"))
print(model)
