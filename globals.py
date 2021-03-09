
"""
This file contains some global switches etc. that we'd like to
use in different parts of the code.
"""

# whether we want to map only from the zero DC mode to others
# or, if False, from any DC mode to any other
ONLY_FROM_ZERO = True

# specify the redshift that we're working at using this switch
# The redshifts are :
#           [0] z = 126
#           [1] z = 3
#           [2] z = 2
#           [3] z = 1
#           [4] z = 0.5
#           [5] z = 0
# TODO we can generalize this into a list and map between redshifts.
#      In that case, redshift has to be an additional style
SNAP_IDX = 5
