import torch
import numpy as np
from models.TSIT_model import TSIT

model = TSIT()
model.netCS(torch.rand(1,3,720,1280))

