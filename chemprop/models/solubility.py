import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def forward_solubility(output: torch.Tensor,temp,mw1,mw2,density2):
