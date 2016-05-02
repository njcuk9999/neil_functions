"""
Description of program
"""
import numpy as np
from scipy.odr import odrpack as odr
from scipy.odr import models
from scipy import interpolate
from astropy.table import Table as table
from astropy.io import fits
import matplotlib.pyplot as plt

# =============================================================================
# Define variables
# =============================================================================


# =============================================================================
# Define classes
# =============================================================================
class spline_return:
    """
    This is the class that controls the use of scipys interpolate splev should only be used via
    the interp1d function
    :param tck: result of interpolate.splrep(x, y, s=0)
    """

    def __init__(self, tck):
        self.tck = tck

    def __call__(self, xnew):
        return interpolate.splev(xnew, self.tck, der=0)


# =============================================================================
# Define functions
# =============================================================================
# Interpolation from Spline function
def interp1d(x, y):
    """
    This is a easier to use spline interpolation
    call by using F = interp1d(oldx, oldy)
    use by using newy = F(oldy)
    :param x: array of floats, must be in numerical order for spline to function
    :param y: array of floats, y values such that y = f(x) to be mapped onto new x values (with a cubic spline)
    :return:
    """
    tck = interpolate.splrep(x, y, s=0)
    sclass = spline_return(tck)
    return sclass

