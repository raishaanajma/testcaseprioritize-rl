import torch
import numpy as np

#this  file is needed to make result reproducible
#it works by giving starting number (seed) so it has consistent result

def seedcode(seed = 63): #starting number
    torch.manual_seed(seed) #to make torch using seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #to make np using seed

seedcode()