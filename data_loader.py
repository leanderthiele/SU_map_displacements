from glob import glob
from enum import Enum, auto
import random

from torch.utils.data.dataset import Dataset as torch_Dataset
from torch.utils.data import DataLoader as torch_DataLoader

import settings
import sim_utils
from simulation_run import SimulationRun
from data_item import DataItem, InputTargetPair

class DataModes(Enum) :
    TRAINING = auto()
    VALIDATION = auto()
    TESTING = auto()

    def __str__(self) :
        if self is DataModes.TRAINING :
            return 'training'
        elif self is DataModes.VALIDATION :
            return 'validation'
        elif self is DataModes.TESTING :
            return 'testing'

class Dataset(torch_Dataset) :
    """
    represents a torch-compatible collection of simulation data
    """
#{{{
    def __init__(self, mode) :
        # populates self.run_pairs, which contains pairs of __Run instances,
        # each pair sorted in such a way that the lower delta_L comes first

        seed_dirs = glob(settings.DATA_PATH+'/seed*')

        self.run_pairs = []
        for seed_dir in seed_dirs :

            run_fnames = glob(seed_dir+'/*[m,p]')

            for i1, run_fname1 in enumerate(run_fnames) :
                run1 = Dataset.SimulationRun(run_fname1)
                if settings.ONLY_FROM_ZERO and not run1.is_zero() :
                    continue

                for i2, run_fname2 in enumerate(run_fnames, start=i1+1) :
                    run2 = Dataset.SimulationRun(run_fname2)
                    self.run_pairs.append(tuple(sorted((run1, run2), key=lambda x : x.delta_L)))

        # we want to randomly shuffle the pairs, but so that each instance does the same shuffling
        random.seed(settings.DATASET_SHUFFLING_SEED)
        random.shuffle(self.run_pairs)
        
        self.mode = mode
        if self.mode is DataModes.TESTING :
            self.run_pairs = self.run_pairs[: settings.NSAMPLES_TESTING]
        elif self.mode is DataModes.VALIDATION :
            self.run_pairs = self.run_pairs[settings.NSAMPLES_TESTING
                                            : settings.NSAMPLES_TESTING+settings.NSAMPLES_VALIDATION]
        elif self.mode is DataModes.TRAINING :
            self.run_pairs = self.run_pairs[settings.NSAMPLES_TESTING+settings.NSAMPLES_VALIDATION :]

    def __getitem__(self, idx) :
        # figure out which run and which data augmentation this idx refers to
        item_idx = idx // 48
        augmentation_index = idx % 48

        # get the normalization functions
        if settings.USE_DENSITY :
            density_norm = settings.DENSITY_NORMALIZATION[self.mode]
        else :
            density_norm = None
        displacement_norm = settings.DISPLACEMENT_NORMALIZATION[self.mode]

        return InputTargetPair(DataItem(self.run_pairs[item_idx][0]),
                               DataItem(self.run_pairs[item_idx][1])) \
                   .normalize(displacement_norm, density_norm) \
                   .augment_data(augmentation_index) \
                   .to_torch()

    def __len__(self) :
        # note that we have 48 augmented vesions for each item!
        return 48 * len(self.run_pairs)
#}}}

class DataLoader(torch_DataLoader) :
    """
    A torch-compatible way to retrieve the data
    """

    def __init__(self, mode) :
        self.dataset = Dataset(mode)
        super().__init__(self.dataset, **settings.DATALOADER_ARGS)
