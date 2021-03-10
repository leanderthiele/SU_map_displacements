import numpy as np
import h5py
import torch

import settings
import sim_utils

class DataItem :
    """
    constructs a network-friendly representation of a simulation run
    """

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

        # we need to channel dimension at the beginning
        # (channel is the 3d dimensionality)
        self.displacement = np.moveaxis(self.displacement, -1, 0)

        if not settings.USE_DENSITY :
            self.tensor = self.displacement
        else :
            self.tensor = np.concatenate((self.density, self.displacement), axis=0)

        self.tensor = torch.as_tensor(self.tensor,
                                      dtype=torch.float32,
                                      device=settings.DEVICE)
