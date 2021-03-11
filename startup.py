"""
Implements a function main() that can be used by importing script.
Its only purpose is to populate the global variables in settings.py
that are not fixed.
"""

import numpy as np
import torch
import torch.multiprocessing as torch_mp

import settings
from network import Network
from data_modes import DataModes

def load_normalizations() :
#{{{
    settings.DENSITY_NORMALIZATION = {}
    settings.DISPLACEMENT_NORMALIZATION = {}

    with np.load(settings.NORMALIZATION_FILE) as f :
        for mode in DataModes :
            sigma_displacement, sigma_density, A, B = f[str(mode)]
            settings.DENSITY_NORMALIZATION[mode] \
                = lambda x : A * ( np.log1p(x/sigma_density) - B )
            settings.DISPLACEMENT_NORMALIZATION[mode] \
                = lambda x : x / sigma_displacement
#}}}

def set_gpu_env() :
#{{{
    settings.NUM_GPU = torch.cuda.device_count()
    settings.DIAGNOSTIC_BARRIER = torch_mp.Barrier(settings.NUM_GPU)
#}}}

def main() :
    load_normalizations()
    set_gpu_env()

    settings.FOO = 'foo'
