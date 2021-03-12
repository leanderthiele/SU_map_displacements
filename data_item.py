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
            # it's useful to have the channel dimension explicit
            self.density = np.expand_dims(self.density, 0)
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

        self.displacement = sim_utils.get_displacement(x1, x2, self.BoxSize)

        # we like to have channel dimension at the beginning
        # (channel is the 3d dimensionality)
        self.displacement = np.moveaxis(self.displacement, -1, 0)

    def normalize(self, mode) :
        # cannot be called after to_torch()
        # TODO in principle, we could think about using the same normalization function
        #      everywhere here. Then it would simply be a collection of magic numbers.

        # load the normalization constants from file and make functions from them
        # NOTE this is quite inefficient (frequent small disk I/O)
        #      but I don't think it's going to be a bottleneck
        with np.load(settings.NORMALIZATION_FILE) as f :
            sigma_displacement, sigma_density, A, B = f[str(mode)]
        density_normalization = lambda x : A * ( np.log1p(x/sigma_density) - B )
        displacement_normalization = lambda x : x / sigma_displacement

        # now call the functions on our numpy arrays
        if settings.USE_DENSITY :
            self.density = density_normalization(self.density)
        self.displacement = displacement_normalization(self.displacement)

        return self

    def __reflect(self, indices) :
        # indices labels the axes that should be reflected

        # do the axis reversal
        # we need to be careful here because of the 0th (channel) dimension
        self.displacement = np.flip(self.displacement, [ii+1 for ii in indices])
        if self.density is not None :
            self.density = np.flip(self.density, [ii+1 for ii in indices])

        # reverse vectorial quantities (displacement components)
        for ii in indices :
            self.displacement[ii, ...] *= -1.0

    def __transpose(self, permutation) :
        # permutation is a permutation of [0,1,2] 

        # when transposing, we need to preserve the channel dimension
        self.displacement = np.transpose(self.displacement, axes=[0,]+[ii+1 for ii in permutation])
        if self.density is not None :
            self.density = np.transpose(self.density, axes=[0,]+[ii+1 for ii in permutation])

        # now we need to interchange the channels (i.e. displacement directions) accordingly,
        self.displacement = self.displacement[permutation, ...]

    def augment_data(self, r) :
        # performs the data augmentation.
        # r should be some integer
        r %= 48
        r1 = r // 6 # this one labels the 8 reflections
        r2 = r % 6  # this one labels the 6 transpositions

        # find the reflection
        reflect_indices = [[], [0,], [1,], [2,], [0,1,], [0,2,], [1,2,], [0,1,2,]][r1]
        if r1 != 0 :
            self.__reflect(reflect_indices)

        # find the transposition
        transpose_permutation = [[0,1,2], [1,0,2], [0,2,1], [1,2,0], [2,0,1], [2,1,0]][r2]
        if r2 != 0 :
            self.__transpose(transpose_permutation)

        return self

    def to_torch(self) :
        # note that this function cannot be called before augment_data
        # TODO do we need to unsqueeze the batch dimension here???
        #      -- I don't think so! pytorch will tell us
        if self.density is None :
            self.tensor = self.displacement
        else :
            self.tensor = np.concatenate((self.density, self.displacement), axis=0)

        self.tensor = torch.as_tensor(self.tensor,
                                      dtype=torch.float32)
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

    def styles(self) :
        # returns the styles describing this InputTargetPair as a torch tensor
        # TODO this is hardcoded at the moment, we may want to change that later
        return torch.tensor([self.item2.delta_L, ], dtype=torch.float32)

    def normalize(self, mode) :
        # performs normalization according to the supplied functions
        # Needs to be called prior to to_torch()
        self.item1 = self.item1.normalize(mode)
        self.item2 = self.item2.normalize(mode)
        return self

    def augment_data(self, rand_int=None) :
        # performs the full set of data augmentations necessary.
        # Note that we have to implement this method here and not in DataItem
        # because we need the transformations to be consistent between input and target
        # rand_int should be a callable that produces a random integer,
        # or a random integer
        if rand_int is None :
            r = np.random.default_rng(hash(self) % 2**32).integers(2**32)
        elif callable(rand_int) :
            r = rand_int()
        elif isinstance(rand_int, int) :
            r = rand_int
        else :
            raise RuntimeError('Invalid argument type of rand_int = {}'.format(rand_int))

        self.item1 = self.item1.augment_data(r)
        self.item2 = self.item2.augment_data(r)

        return self

    def to_torch(self) :
        # note that this function cannot be called before augment_data,
        self.item1 = self.item1.to_torch()
        self.item2 = self.item2.to_torch()
        return self
#}}}
