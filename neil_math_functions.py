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


def rchisquared(x, y, ey, model, p):
    """
    Calculates the reduced chisquared value based on x and y and a model
    :param x: [numpy array] x axis data (data to base model on)
    :param y: [numpy array] y axis data (data to fit model to)
    :param ey: [numpy array] uncertainties in y axis data
    :param model: [function] function taking arguements x and *p
                             i.e. lambda x, *p: p[0]*x + p[1]*x
    :param p: [list] list of parameters for model for a model with 2 fit
                     parameters p = [a, b] and function would require
                     x, a, b as arguments
    :return: reduced chi squared, degrees of freedom (N - n - 1)
    """
    chi2 = np.sum(pow(((y - model(p, x)) / ey), 2))
    DOF = len(y) - len(p) - 1
    rchi2 = chi2 / float(DOF)
    return rchi2, DOF


def chisquare(y1, y2, ey1):
    """
    calculate chi squared for a data set (with uncertainties) and a model
    :param y1: array data
    :param y2: array model
    :param ey1: array uncertainties on data
    :return:
    """
    return np.sum(((y1 - y2) / ey1) ** 2)
