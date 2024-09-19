import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def forward_solubility(Delta_G_fus,Gamma,temp,mw1,mw2,density2):
    xi=torch.exp(-1*Delta_G_fus)/(Gamma*temp)
    solvent_density=density2/1000
    moles_solvent=solvent_density/(mw2/1000)
    s=(xi*moles_solvent)/(1-xi)
    log_s=torch.log(s)
    return log_s
