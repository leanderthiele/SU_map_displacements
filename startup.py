import os.path
import warnings
import argparse
import numpy as np

import settings
from settings import ToSet
import data_loader

class ArgParser :
    """
    upon construction, this will parse the command line.
    Add arguments that should be parsed to the `populate_parser' method.
    """
#{{{
    def populate_parser(self) :
        self.parser.add_argument('--id', nargs='?', default='debug',
                                 help='string that identifies data associated with this run')
    def __init__(self) :
        # construct the argument parser
        self.parser = argparse.ArgumentParser()

        # load the switches we would like to parse
        self.populate_parser()

    def parse_and_set(self) :
        args = self.parser.parse_args()

        if args.id == 'debug' :
            warnings.warn('--id not set, defaulting to debug')
        settings.ID = settings.ID.set(args.id)
#}}}

def populate_settings_from_cl() :
    """
    populates some settings that can be done directly from the command line
    """
#{{{
    args = ArgParser()
    args.parse_and_set()
#}}}

def load_normalizations() :
    """
    populates the normalization function dicts
    """
#{{{
    settings.DENSITY_NORMALIZATIONS \
        = settings.DENSITY_NORMALIZATIONS.set(dict())
    settings.DISPLACEMENT_NORMALIZATIONS \
        = settings.DISPLACEMENT_NORMALIZATIONS.set(dict())
    with np.load(settings.NORMALIZATION_FILE) as f :
        for mode in data_loader.DataModes :
            sigma_displacement, sigma_density, A, B = f[str(mode)]
            settings.DENSITY_NORMALIZATIONS[mode] \
                = lambda x : A * ( np.log1p(x/sigma_density) - B )
            settings.DISPLACEMENT_NORMALIZATIONS[mode] \
                = lambda x : x / sigma_displacement
#}}}

def set_filenames() :
    """
    populates the input/output filenames
    """
#{{{
    settings.LOSS_FILE = settings.LOSS_FILE.set(os.path.join(settings.RESULTS_PATH, 'loss_%s.npz'%settings.ID))
    settings.MODEL_FILE = settings.MODEL_FILE.set(os.path.join(settings.RESULTS_PATH, 'model_%s.pt'%settings.ID))
#}}}

def check_all_set() :
    """
    checks that none of the variables in the settings module is still ToSet.
    (or, if it is a composite type like a list, tuple, dict, class, etc.,
     whether any of its members are still ToSet)
    This is a good way to check that startup.main() has done its job.
    """
#{{{
    for name in dir(settings) :
        if name.startswith('__') :
            # this is an internal variable that we don't care about
            continue
        this_setting = settings.__dict__[name]

        if isinstance(this_setting, ToSet) :
            raise RuntimeError('settings.%s not set by startup.main'%name)

        try :
            # dict-like
            for k, v in this_setting.items() :
                if isinstance(v, ToSet) :
                    raise RuntimeError('settings.%s[%s] not set by startup.main'%(name, k))
        except AttributeError :
            pass

        try :
            # iterable
            for ii, v in enumerate(this_setting) :
                if isinstance(v, ToSet) :
                    raise RuntimeError('settings.%s[%d] not set by startup.main'%(name, ii))
        except TypeError :
            pass

        try :
            # custom class
            for k, v in vars(this_setting).items() :
                if k.startswith('__') :
                    continue
                if isinstance(v, ToSet) :
                    raise RuntimeError('settings.%s.%s not set by startup.main'%(name, k))
        except TypeError :
            pass
#}}}

def main(mode) :
    """
    this is the only function that should be called from this module.
    `mode' refers to the global mode of execution.
    """
    assert isinstance(mode, data_loader.DataModes)

    # tell anyone using the settings module that it
    # is in the correct state
    # (all global variables have been updated)
    settings.STARTUP_CALLED = True

    # this should be called right at the beginning so subsequent
    # action can make use of this information if necessary
    # and we don't have to pass the `mode' argument around
    settings.MODE = settings.MODE.set(mode)

    populate_settings_from_cl()
    set_filenames()
    load_normalizations()
    check_all_set()
