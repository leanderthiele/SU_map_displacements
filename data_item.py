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
    def __init__(self, mode, run) :

        self.mode = mode

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

        self.tensor = None
        self.is_normalized = False
        self.is_augmented = False

    def normalize_for_input(self) :
        # cannot be called after to_torch()
        # mutually exclusive with normalize_for_target

        # TODO in principle, we could think about using the same normalization function
        #      everywhere here. Then it would simply be a collection of magic numbers.
        # TODO we also need to normalize the overdensity

        assert self.tensor is None
        assert not self.is_normalized

        if settings.USE_DENSITY :
            self.density = settings.DENSITY_NORMALIZATIONS[self.mode](self.density)
        self.displacement = settings.DISPLACEMENT_NORMALIZATIONS[self.mode](self.displacement)

        self.delta_L = settings.DELTA_L_NORMALIZATIONS[self.mode](self.delta_L)

        self.is_normalized = True

        return self

    def normalize_for_target(self) :
        # cannot be called after to_torch()
        # mutually exclusive with normalize_for_input
        
        assert self.tensor is None
        assert not self.is_normalized

        # normalize such that the output is in the range [ -1, 1 ]
        self.displacement /= 0.5 * settings.MAX_BOX_SIZE
        
        # Note that delta_L is actually used as an input, but this structure
        # should be clear
        self.delta_L = settings.DELTA_L_NORMALIZATIONS[self.mode](self.delta_L)

        self.is_normalized = True

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
        permutationp1 = [0,] + [ii+1 for ii in permutation]
        self.displacement = np.transpose(self.displacement, axes=permutationp1)
        if self.density is not None :
            self.density = np.transpose(self.density, axes=permutationp1)

        # now we need to interchange the channels (i.e. displacement directions) accordingly,
        self.displacement = self.displacement[permutation, ...]

    def augment_data(self, r) :
        # cannot be called after to_torch() or normalize
        # performs the data augmentation.
        # r should be some integer

        assert self.tensor is None
        assert not self.is_normalized
        assert not self.is_augmented

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

        self.is_augmented = True

        return self

    def to_torch(self) :
        # call this method right at the end, after augmentation and normalization

        assert self.is_normalized
        assert self.is_augmented

        if self.density is None :
            # reflections etc. can induce negative strides in the displacement field,
            # which torch.as_tensor() cannot handle.
            # Thus, we make a copy.
            # Note that in the other case (self.density is not None), the concatenation
            # produces a contiguous copy automatically
            self.tensor = self.displacement.copy()
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

        assert self.item2.is_normalized

        return torch.tensor([self.item2.delta_L, ], dtype=torch.float32)

    def normalize(self) :
        # performs normalization according to the supplied functions
        # Needs to be called prior to to_torch()
        self.item1 = self.item1.normalize_for_input()
        self.item2 = self.item2.normalize_for_target()
        return self

    def augment_data(self, rand_int=None) :
        # performs the full set of data augmentations necessary.
        # Note that we have to implement this method here and not in DataItem
        # because we need the transformations to be consistent between input and target
        # rand_int should be a callable that produces a random integer,
        # or a random integer
        
        if rand_int is None :
            r = np.random.default_rng(id(self) % 2**32).integers(2**32)
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
