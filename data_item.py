import numpy as np
import h5py
import torch

import settings
import sim_utils

class DataItem :
    """
    constructs a network-friendly representation of a simulation run
    """
#{{{
    def __init__(self, run) :

        self.seed = run.seed
        self.delta_L = run.delta_L

        if settings.USE_DENSITY :
            with np.load(run.density_fname()) as f :
                self.density = f['density']
                if not settings.H_UNITS :
                    self.density *= f['h']**2
        else :
            self.density = None

        snap1_fname = run.reference_snap_fname()
        snap2_fname = run.snap_fname()

        x1 = sim_utils.load_particles(snap1_fname)
        x2 = sim_utils.load_particles(snap2_fname)

        with h5py.File(snap2_fname, 'r') as f :
            self.HubbleParam = f['Parameters'].attrs['HubbleParam']
            self.BoxSize = f['Header'].attrs['BoxSize']

        if not settings.H_UNITS :
            x1 /= self.HubbleParam
            x2 /= self.HubbleParam
            self.BoxSize /= self.HubbleParam

        self.displacement = sim_utils.get_displacement(x1, x2, BoxSize)

        # we need to have channel dimension at the beginning
        # (channel is the 3d dimensionality)
        self.displacement = np.moveaxis(self.displacement, -1, 0)

        if not settings.USE_DENSITY :
            self.tensor = self.displacement
        else :
            self.tensor = np.concatenate((self.density, self.displacement), axis=0)

    def normalize(self, displacement_norm, density_norm) :
        # cannot be called after to_torch()
        if settings.USE_DENSITY :
            assert density_norm is not None
            self.tensor[0, ...] = density_norm(self.tensor[0, ...])
        else :
            assert density_norm is None

        offset = 1 if settings.USE_DENSITY else 0
        self.tensor[offset:, ...] = displacement_norm(self.tensor[offset:, ...])

        return self

    def __reflect(self, indices) :
        # indices labels the axes that should be reflected

        # do the axis reversal
        self.tensor = np.flip(self.tensor, indices)

        # reverse vectorial quantities (displacement components)
        offset = 1 if settings.USE_DENSITY else 0
        for ii in indices :
            self.tensor[offset+ii, ...] *= -1.0

    def __transpose(self, permutation) :
        # permutation is a permutation of [0,1,2] 

        # when transposing, we need to preserve the channel dimension
        self.tensor = np.transpose(self.tensor, axes=[0,]+[ii+1 for ii in permutation])

        # now we need to interchange the channels (i.e. displacement directions) accordingly,
        # keeping the density channel in the zeroth position if it is present
        if settings.USE_DENSITY :
            indices = [0,]+[ii+1 for ii in permutation]
        else :
            indices = permutation
        self.tensor = self.tensor[indices, ...]

    def augment_data(self, r) :
        # performs the data augmentation.
        # r should be some integer
        r %= 48
        r1 = r // 6 # this one labels the 8 reflections
        r2 = r % 6  # this one labels the 6 transpositions

        # find the reflection
        reflect_indices = [[,], [0,], [1,], [2,], [0,1,], [0,2,], [1,2,], [0,1,2,]][r1]
        if r1 != 0 :
            self.__reflect(reflect_indices)

        # find the transposition
        transpose_permutation = [[0,1,2], [1,0,2], [0,2,1], [1,2,0], [2,0,1], [2,1,0]][r2]
        if r2 != 0 :
            self.__transpose(transpose_permutation)

        return self

    def to_torch(self) :
        # note that this function cannot be called before augment_data,
        # but not after pin_memory!
        # TODO do we need to unsqueeze the batch dimension here???
        self.tensor = torch.as_tensor(self.tensor,
                                      dtype=torch.float32,
                                      device=settings.DEVICE)
        return self
        
    def pin_memory(self) :
        # according to PyTorch documentation,
        # we should define this method on custom types returned by the data loader
        self.tensor = self.tensor.pin_memory()
        return self
#}}}

class InputTargetPair :
    """
    simply a wrapper around two DataItem's
    """
#{{{
    def __init__(self, item1, item2) :
        self.item1 = item1
        self.item2 = item2

    def normalize(self, displacement_norm, density_norm) :
        # performs normalization according to the supplied functions
        # Needs to be called prior to to_torch()
        self.item1 = self.item1.normalize(displacement_norm, density_norm)
        self.item2 = self.item2.normalize(displacement_norm, density_norm)
        return self

    def augment_data(self, rand_int=None) :
        # performs the full set of data augmentations necessary.
        # Note that we have to implement this method here and not in DataItem
        # because we need the transformations to be consistent between input and target
        # rand_int should be a callable that produces a random integer,
        # or a random integer
        if rand_int is None :
            r = np.random.default_rng(hash(self) % 2**32).integers(2**32)
        else if isinstance(rand_int, callable) :
            r = rand_int()
        else if isinstance(rand_int, int) :
            r = rand_int
        else :
            raise RuntimeError('Invalid argument type of rand_int = {}'.format(rand_int))

        self.item1 = self.item1.augment_data(r)
        self.item2 = self.item2.augment_data(r)

        return self

    def to_torch(self) :
        # note that this function cannot be called before augment_data,
        # but not after pin_memory!
        self.item1 = self.item1.to_torch()
        self.item2 = self.item2.to_torch()
        return self

    def pin_memory(self) :
        # according to PyTorch documentation,
        # we should define this method on custom types returned by the data loader
        self.item1 = self.item1.pin_memory()
        self.item2 = self.item2.pin_memory()
        return self
#}}}
