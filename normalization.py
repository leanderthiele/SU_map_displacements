"""
A small script to compute the coefficients needed in our normalization functions.
"""


import math
import numpy as np

import settings
from data_loader import DataModes
from simulation_run import get_runs
from data_item import DataItem

# FIXME
#
# This script is currently broken, because it cannot be run
# on a CPU-only node.
# The reason is that the Dataset class uses some of the variables
# that are set during startup.main(), which we definitely don't want
# to call here.
# 
# A possible solution would be to have optional `rank', `world_size'
# arguments in Dataset (or simply `for_normalization')


DENSITY_FACTOR = 10.0


def normalization(mode) :
    """
    We normalize using the functions :

                                 displacement
        displacement <----   -------------------
                             sigma(displacement)


                        {      (                  density            )        }
        density <---- A {  log ( 1 + ------------------------------- )  -  B  }
                        {      (     DENSITY_FACTOR * sigma(density) )        }

    where A, B are chosen such that the transformed density field as unit variance and zero mean

    The output is a list containing :
        { sigma(displacement), DENSITY_FACTOR * sigma(density), A, B }
    """
#{{{
    assert settings.USE_DENSITY # for convienience

    run_pairs = get_runs(mode, use_bounds=False) # we do this in a single process, so we set rank and world_size to 0, 1

    # make a first pass through the data, computing the variances
    # note that we're using the fact that mean(displacement) ~ 0
    # and mean(density) = const across boxes, so we don't have to carry a global mean
    # between different boxes for either
    var_displacement = 0.0
    var_density = 0.0

    for run_pair in run_pairs :
        print('In normalization.py, first loop, %s'%str(run_pair[0]))
        item = DataItem(mode, run_pair[0]) # we only want to normalize the input
        var_displacement += np.var(item.displacement)
        var_density += np.var(item.density)

    var_displacement /= len(run_pairs)
    var_density /= len(run_pairs)

    # make a second pass through the data, now computing the variance and mean
    # of the already transformed density field
    # this is super inefficient but we only need to do this once hopefully
    # here, we cannot rely on the simplifications afforded in the previous run,
    # so we need to compute the variance carefully
    var_logdensity = 0.0
    avg_logdensity = 0.0

    for run_pair in run_pairs :
        print('In normalization.py, second loop, %s'%str(run_pair[0]))
        item = DataItem(mode, run_pair[0])
        logdensity = np.log1p( item.density / DENSITY_FACTOR / math.sqrt(var_density) )
        var_logdensity += np.sum(logdensity**2) / logdensity.size
        avg_logdensity += np.mean(logdensity)

    avg_logdensity /= len(run_pairs)
    var_logdensity /= len(run_pairs)
    var_logdensity -= avg_logdensity**2

    return [ math.sqrt(var_displacement),
             DENSITY_FACTOR * math.sqrt(var_density), 
             1.0 / math.sqrt(var_logdensity),
             avg_logdensity, ]
#}}}

if __name__ == '__main__' :
    norms = {}
    for mode in DataModes :
        norms[str(mode)] = np.array(normalization(mode))
    np.savez(settings.NORMALIZATION_FILE, **norms,
             README=normalization.__doc__+'\n\nThis normalization is for files in '+settings.DATA_PATH)
