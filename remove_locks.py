"""
A simple helper script, to be called from outside any multiprocessing (!),
that removes all the lock files at the start of a run to make sure any abnormal
termination in previous runs does not leave us with stale locks.
"""


from glob import glob
import os
import os.path
from time import time
import subprocess
import system

import settings

def is_old_file(fname) :
#{{{
    # in seconds
    diff = time() - os.stat(fname).st_mtime

    return diff > settings.SLURM_TIME_DIFF * 60
#}}}


def has_abort_msg(fname) :
#{{{
    last_lines = subprocess.getoutput('tail -n 5 %s'%fname)
    indicators = ['slurmstepd', 'Error']
    return any([indicator in last_lines for indicator in indicators])
#}}}


def other_slurm_job_running() :
    """
    returns true if another already running slurm job is found that looks like it is also
    from this project.
    returns false otherwise
    """
#{{{

    # look for slurm output files
    slurm_files = glob('slurm-*.out')

    try :
        slurm_files.remove('slurm-%s.out'%os.environ['SLURM_JOB_ID'])
    except ValueError :
        raise RuntimeError('Unable to find the slurm output file this job is writing to!')

    # loop through the remaining slurm output files and see if there is one that
    # is currently being written to -- we take this as indication
    # that there's another job running
    for slurm_file in slurm_files :
        if not ( has_abort_msg(slurm_file) or is_old_file(slurm_file) ) :
            print('Found active slurm job besides ourselves writing to %s'%slurm_file)
            return True

    print('Did not find active slurm job besides ourselves')
    return False

#}}}

def remove_lock_file(fname) :
    """
    removes the lock file called fname
    We want to be really really sure this is really a lock file, so we do some checks
    to make sure we do not accidentally delete data
    """
#{{{

    # check that argument is a string
    if not isinstance(fname, str) :
        raise RuntimeError(f'{fname} is not a string')

    # check that extension is as expected
    _, ext = os.path.splitext(fname)
    if not ext == settings.LOCK_EXTENSION :
        raise RuntimeError(f'{fname} does not have the lock file extension')

    # check that the file actually exists
    if not os.path.isfile(fname) :
        raise RuntimeError(f'{fname} does not exist')

    # get file properties -- currently we are only using the size attribute of this object
    fstats = os.stat(fname)

    # check that file size is zero
    if not fstats.st_size == 0 :
        raise RuntimeError(f'{fname} has non-zero size')

    # we have passed all the checks, now it should be safe to remove the file
    os.remove(fname)

#}}}


def remove_lock_files() :
    """
    Does what the name says.
    Currently, it deals with the snap hdf5 locks, the density npz locks, and the normalization npz lock
    """
#{{{

    # first we remove the locks from the hdf5 snap files and npz density field files
    seed_dirs = glob(settings.DATA_PATH+'/seed*'))
    for seed_dir in seed_dirs :
        run_dirs = glob(seed_dir+'/*[m,p]')
        for run_dir in run_dirs :
            snap_locks = glob('%s.%s'%(run_dir+'/G4/snap_*.hdf5', settings.LOCK_EXTENSION))
            density_locks = glob('%s.%s'%(run_dir+'/G4/snap_*_postprocessing/density_%d.npz'%settings.NSIDE,
                                          settings.LOCK_EXTENSION))
            for sl in snap_locks :
                remove_lock_file(sl)
            for dl in density_locks :
                remove_lock_file(dl)

    # now we remove the lock on the normalization file
    nl = '%s.%s'%(settings.NORMALIZATION_FILE, settings.LOCK_EXTENSION)
    if os.path.isfile(nl) :
        remove_lock_file(nl)

#}}}

def main() :
    """
    will be called if this file is executed.

    First checks if there is another slurm job running that looks like it is also executing
    some code from this project. In that case no action is taken
    (it is then assumed that this job already handled the removal of potentially stale lock files,
     and we definitely don't want to interfere with what this job is currently doing).

    Otherwise, removes all existing lock files.
    """
#{{{

    if settings.STARTUP_CALLED :
        raise RuntimeError('settings.STARTUP_CALLED is True')

    if not other_slurm_job_running() :
        print('No other active slurm job identified, will remove lock files')
        remove_lock_files()
    else :
        print('Found another active slurm job, will not remove lock files')

#}}}


if __name__ == '__main__' :
    main()
