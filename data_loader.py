"""
Wrappers around the torch Dataset and DataLoader classes.
"""


from copy import copy
from glob import glob
from enum import Enum, auto
from time import time
import numpy as np

import torch
from torch.utils.data.dataset import Dataset as torch_Dataset
from torch.utils.data import DataLoader as torch_DataLoader

import settings
import startup
import sim_utils
from simulation_run import get_runs
from data_item import DataItem, InputTargetPair

class DataModes(Enum) :
    """
    the three possible modes for which we may want to load data
    """
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
        """
        populates self.run_pairs, which contains pairs of SimulationRun instances,
        each pair sorted in such a way that the lower delta_L comes first
        """

        self.mode = mode 

        self.run_pairs = get_runs(self.mode, deltaLbounds=settings.DELTA_L_BOUNDS)

        if settings.RANK == 0 :
            print('%d samples in the %s set'%(len(self.run_pairs), str(mode)))

    def getitem_all(self, idx) :
        """
        operates on the entire dataset, i.e. idx is within the whole set of simulations
        and augmentations
        """

        # figure out which run and which data augmentation this idx refers to
        run_idx = idx // settings.N_AUGMENTATIONS

        if settings.N_AUGMENTATIONS != 48 :
            # need to use a non-deterministic method, otherwise we won't
            # sample all augmentations
            augmentation_index = np.random.randint(48)
        else :
            # can use the deterministic method, sampling all augmentations
            augmentation_index = idx % settings.N_AUGMENTATIONS

        return InputTargetPair(DataItem(self.mode, self.run_pairs[run_idx][0]),
                               DataItem(self.mode, self.run_pairs[run_idx][1])) \
                   .augment_data(augmentation_index) \
                   .normalize() \
                   .to_torch()

    def __getitem__(self, idx) :
        """
        operates on the sub-dataset corresponding to this rank
        """

        return self.getitem_all(idx * settings.WORLD_SIZE + settings.RANK)

    def len_all(self) :
        """
        operates on the entire dataset
        """

        return settings.N_AUGMENTATIONS * len(self.run_pairs)

    def __len__(self) :
        """
        operates on the sub-dataset corresponding to this rank
        """

        return self.len_all() // settings.WORLD_SIZE
#}}}

class Batch :
    """
    represents a Batch of data items

    Upon construction, Batch has the fields
        inputs, targets, guesses, styles,
    each with the 0th dimension the batch dimension

    We expose the method get_on_device() that returns the tuple
        inputs, targets, guesses, styles
    on the device local to the process
    """
#{{{
    def __init__(self, data_items) :

        # we use this method as collate_fn, in which case data_items
        # will be a list of InputTargetPair's, or, if automatic batching is disabled,
        # a single InputTargetPair
        if isinstance(data_items, list) :
            assert isinstance(data_items[0], InputTargetPair)
        else :
            assert isinstance(data_items, InputTargetPair)
            # convert into list so the rest of the code is invariant
            data_items = [data_items, ]

        Nchannels = 4 if settings.USE_DENSITY else 3

        # NOTE in the following, we require pin_memory = False.
        #      the reason is that otherwise we need to establish a CUDA context for each
        #      worker process, which carries ~ 600 MB that we cannot afford.
        #      The data copying is fast enough anyway, so no worries here.
        self.inputs = torch.empty(len(data_items), Nchannels, *[settings.NSIDE,]*3,
                                  device=torch.device('cpu'),
                                  pin_memory=False,
                                  dtype=torch.float32)
        self.targets = torch.empty(len(data_items), 3, *[settings.NSIDE,]*3,
                                   device=torch.device('cpu'),
                                   pin_memory=False,
                                   dtype=torch.float32)
        self.guesses = torch.empty(len(data_items), 3, *[settings.NSIDE,]*3,
                                   device=torch.device('cpu'),
                                   pin_memory=False,
                                   dtype=torch.float32)
        self.styles = torch.empty(len(data_items), settings.NSTYLES,
                                  device=torch.device('cpu'),
                                  pin_memory=False,
                                  dtype=torch.float32)

        # for the target, we only require the displacement field,
        # so we throw the density away
        offset = 1 if settings.USE_DENSITY else 0

        # collect the batch
        for ii, data_item in enumerate(data_items) :
            self.inputs[ii, ...] = data_item.item1.tensor
            self.targets[ii, ...] = data_item.item2.tensor[offset:, ...]
            self.guesses[ii, ...] = data_item.item1.guess
            self.styles[ii, ...] = data_item.styles()

        # sanity check
        assert torch.min(self.targets).item() >= -0.5
        assert torch.max(self.targets).item() <= +0.5
        assert torch.min(self.guesses).item() >= -0.5
        assert torch.max(self.guesses).item() <= +0.5


    def get_on_device(self) :
        return self.inputs.to(settings.DEVICE_IDX), \
               self.targets.to(settings.DEVICE_IDX), \
               self.guesses.to(settings.DEVICE_IDX), \
               self.styles.to(settings.DEVICE_IDX)

#}}}

class WorkerPool :
    """
    at the moment, this is simply used as a callable for the worker_init_fn
    argument for the pytorch DataLoader in order to do stuff that we require
    to happen at the start of a worker process.
    """
#{{{
    def __init__(self, mode) :
        self.mode = mode

        # we need to store the process-dependent stuff locally
        # because we don't have it available in settings when init_worker is called
        # the copy is superfluous at the moment because integers are not mutable,
        # but we never know if we may want to change the types later (improbable though)
        self.rank = copy(settings.LOCAL_RANK)

    def init_worker(self, worker_id) :
        """
        use this method as worker_init_fn
        """

        # since this is called in a separate process,
        # we need to get a consistent view of the settings
        startup.main(self.mode, self.rank)

        # initialize the random seed for this process
        # we don't use just the worker_id but also the rank
        # so we truly get different random numbers in all workers,
        # not restricted to the current pool
        # note that we get some entropy from the time
        # so different epochs get different data augmentations
        np.random.seed((hash(time())
                        + (settings.RANK * torch.utils.data.get_worker_info().num_workers
                           + worker_id)) % 2**32)
#}}}

class DataLoader(torch_DataLoader) :
    """
    A torch-compatible way to retrieve the data
    """
#{{{
    def __init__(self, mode) :
        self.dataset = Dataset(mode)
        self.worker_pool = WorkerPool(mode)
        super().__init__(self.dataset,
                         collate_fn=Batch,
                         # the workers are implemented as separate processes,
                         # so we need to make sure they seed a consistent view of the configuration options
                         worker_init_fn=self.worker_pool.init_worker,
                         **settings.DATALOADER_ARGS)
#}}}
