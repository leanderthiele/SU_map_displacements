from glob import glob
from enum import Enum, auto
import numpy as np

import torch
from torch.utils.data.dataset import Dataset as torch_Dataset
from torch.utils.data import DataLoader as torch_DataLoader

import settings
import startup
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
    def __init__(self, mode, rank, world_size) :
        # populates self.run_pairs, which contains pairs of __Run instances,
        # each pair sorted in such a way that the lower delta_L comes first

        seed_dirs = glob(settings.DATA_PATH+'/seed*')

        self.run_pairs = []
        for seed_dir in seed_dirs :

            run_fnames = glob(seed_dir+'/*[m,p]')

            for i1, run_fname1 in enumerate(run_fnames) :
                run1 = SimulationRun(run_fname1)
                if settings.ONLY_FROM_ZERO and not run1.is_zero() :
                    continue

                for i2, run_fname2 in enumerate(run_fnames[i1+1:]) :
                    run2 = SimulationRun(run_fname2)
                    self.run_pairs.append(tuple(sorted((run1, run2), key=lambda x : x.delta_L)))

        # we want to randomly shuffle the pairs, but so that each instance does the same shuffling
        np.random.default_rng(settings.DATASET_SHUFFLING_SEED).shuffle(self.run_pairs)
        
        self.mode = mode
        if self.mode is DataModes.TESTING :
            self.run_pairs = self.run_pairs[: settings.NSAMPLES_TESTING]
        elif self.mode is DataModes.VALIDATION :
            self.run_pairs = self.run_pairs[settings.NSAMPLES_TESTING
                                            : settings.NSAMPLES_TESTING+settings.NSAMPLES_VALIDATION]
        elif self.mode is DataModes.TRAINING :
            self.run_pairs = self.run_pairs[settings.NSAMPLES_TESTING+settings.NSAMPLES_VALIDATION :]
        else :
            raise RuntimeError('Invalid mode {}'.format(self.mode))

        self.rank = rank
        self.world_size = world_size

    def getitem_all(self, idx) :
        # operates on the entire dataset

        # figure out which run and which data augmentation this idx refers to
        item_idx = idx // 48
        augmentation_index = idx % 48

        return InputTargetPair(DataItem(self.mode, self.run_pairs[item_idx][0]),
                               DataItem(self.mode, self.run_pairs[item_idx][1])) \
                   .normalize() \
                   .augment_data(augmentation_index) \
                   .to_torch()

    def __getitem__(self, idx) :
        # operates on the sub-dataset corresponding to this rank
        return self.getitem_all(idx * self.world_size + self.rank)

    def len_all(self) :
        # operates on the entire dataset

        # note that we have 48 augmented vesions for each item!
        return 48 * len(self.run_pairs)

    def __len__(self) :
        # operates on the sub-dataset corresponding to this rank
        return self.len_all() // self.world_size
#}}}

class Batch :
    """
    represents a Batch of data items
    (we use the constructor to extract the required fields)
    The constructed Batch has the fields
        inputs, output, styles,
    each with the 0th dimension the batch dimension
    """
#{{{
    def __init__(self, device) :
        self.device = device
        self.inputs = None
        self.targets = None
        self.styles = None

    def __call__(self, data_items) :
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
        self.inputs = torch.empty(len(data_items), Nchannels, *[settings.NSIDE,]*3,
                                  device=torch.device('cpu'),
                                  pin_memory=False, # TODO
                                  dtype=torch.float32)
        self.targets = torch.empty(len(data_items), 3, *[settings.NSIDE,]*3,
                                   device=torch.device('cpu'),
                                   pin_memory=False, # TODO
                                   dtype=torch.float32)
        self.styles = torch.empty(len(data_items), settings.NSTYLES,
                                  device=torch.device('cpu'),
                                  pin_memory=False, # TODO
                                  dtype=torch.float32)

        # for the target, we only require the displacement field,
        # so we throw the density away
        offset = 1 if settings.USE_DENSITY else 0

        # collect the batch
        for ii, data_item in enumerate(data_items) :
            self.inputs[ii, ...] = data_item.item1.tensor
            self.targets[ii, ...] = data_item.item2.tensor[offset:, ...]
            self.styles[ii, ...] = data_item.styles()

        # now push the batch to the desired device
        self.inputs = self.inputs.to(self.device, non_blocking=True)
        self.targets = self.targets.to(self.device, non_blocking=True)
        self.styles = self.styles.to(self.device)

        return self
#}}}

class WorkerPool :
    """
    at the moment, this is simply used as a callable for the worker_init_fn
    argument for the pytorch DataLoader in order to do stuff that we require
    to happen at the start of a worker process.

    Note that `rank' and `world_size' refer to the top-level multiprocessing,
    not the lower level multiprocessing that happens in the worker teams.
    """
#{{{
    def __init__(self, mode, rank, world_size) :
        self.mode = mode
        self.rank = rank
        self.world_size = world_size

    def init_worker(self, worker_id) :
        # use this method as worker_init_fn

        # since this is called in a separate process,
        # we need to get a consistent view of the settings
        startup.main(self.mode)
#}}}

class DataLoader(torch_DataLoader) :
    """
    A torch-compatible way to retrieve the data
    """
#{{{
    def __init__(self, mode, rank, world_size) :
        self.dataset = Dataset(mode, rank, world_size)
        self.worker_pool = WorkerPool(mode, rank, world_size)
        super().__init__(self.dataset,
                         collate_fn=Batch(rank),
                         # the workers are implemented as separate processes,
                         # so we need to make sure they seed a consistent view of the configuration options
                         worker_init_fn=self.worker_pool.init_worker,
                         **settings.DATALOADER_ARGS)
#}}}
