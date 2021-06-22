import os.path
from glob import glob
import numpy as np

import settings
import data_loader # import whole module to avoid circular import

class SimulationRun :
    """
    represents a single simulation run and implements my canonical directory structure
    """
#{{{
    def __init__(self, dirname) :
        self.dirname = dirname
        seed_root, run_root = os.path.split(os.path.normpath(self.dirname).rstrip('/'))

        self.seed = int(seed_root.split('seed')[-1])

        self.delta_L = float(run_root.strip('p').strip('m'))
        if run_root[-1] == 'm' :
            self.delta_L *= -1.0
        else :
            assert run_root[-1] == 'p'

    def __repr__(self) :
        return self.dirname

    def snap_fname(self) :
        # returns the filename of the hdf5 particle catalog
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d.hdf5'%settings.SNAP_IDX)

    def reference_snap_fname(self) :
        # returns the filename of the high-redshift hdf5 particle catalog
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d.hdf5'%0)

    def density_fname(self) :
        # returns the filename of the density field
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d_postprocessing'%settings.SNAP_IDX,
                            'density_%d.npz'%settings.NSIDE)

    def is_zero(self) :
        # returns whether this is a zero delta_L run
        return abs(self.delta_L) < 1e-10
#}}}


def get_runs(mode, use_bounds=True) :
    """
    returns a list of pairs of SimulationRun instances, corresponding to the current mode
    """
#{{{
    seed_dirs = glob(settings.DATA_PATH+'/seed*')

    # to exclude any possible correlation
    np.random.default_rng(settings.DATASET_SHUFFLING_SEED).shuffle(seed_dirs)

    # choose the simulation seeds that we want to use in this mode
    if mode is data_loader.DataModes.TESTING :
        seed_indices = slice(0, settings.NSEEDS_TESTING)
    elif mode is data_loader.DataModes.VALIDATION :
        seed_indices = slice(settings.NSEEDS_TESTING, settings.NSEEDS_TESTING+settings.NSEEDS_VALIDATION)
    elif mode is data_loader.DataModes.TRAINING :
        seed_indices = slice(settings.NSEEDS_TESTING+settings.NSEEDS_VALIDATION, None)
    else :
        raise RuntimeError('Invalid mode {}'.format(mode))
    seed_dirs = seed_dirs[seed_indices]

    run_pairs = []
    for seed_dir in seed_dirs :

        run_fnames = glob(seed_dir+'/*[m,p]')

        for i1, run_fname1 in enumerate(run_fnames) :
            run1 = SimulationRun(run_fname1)

            if settings.ONLY_FROM_ZERO and not run1.is_zero() :
                continue

            for i2, run_fname2 in enumerate(run_fnames[i1+1:]) :
                run2 = SimulationRun(run_fname2)

                if use_bounds :
                    # check if we want to include this value of the overdensity
                    if run2.delta_L < settings.DELTA_L_BOUNDS[0] \
                       or run2.delta_L > settings.DELTA_L_BOUNDS[1] :
                        continue

                    if not settings.ONLY_FROM_ZERO :
                        if run1.delta_L < settings.DELTA_L_BOUNDS[0] \
                           or run1.delta_L > settings.DELTA_L_BOUNDS[1] :
                            continue

                # all checks passed --> append to the output
                run_pairs.append(tuple(sorted((run1, run2), key=lambda x : x.delta_L)))

    # we want to randomly shuffle the pairs, but so that each instance does the same shuffling
    # (shuffling removes correlations, for example we don't want to train on simulations
    #  with the same seed consecutively)
    np.random.default_rng(settings.DATASET_SHUFFLING_SEED).shuffle(run_pairs)

    return run_pairs
#}}}
