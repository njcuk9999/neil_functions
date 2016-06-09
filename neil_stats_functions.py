import numpy as np
from scipy.special import erf, erfinv


# =============================================================================
# Define variables
# =============================================================================
sqrt2 = np.sqrt(2)
sqrt2pi = np.sqrt(2 * np.pi)


# =============================================================================
# Define functions
# =============================================================================
def gaussian1d_variant1(x, *p):
    """
    Create a normalised gaussian (Area == 1) from mean and variance

    :param x: [numpy array]    x axis data (array/list)
    :param p: [Tuple]         (mean , variance)

    Area of gaussian is 1  --> A = 1/(c*sqrt(2pi))

    Returns a Gaussian array one value for each x value
    """
    mu, sigma = p
    A = 1.0 / (sigma * sqrt2pi)
    return A * np.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))


def gaussian1d_variant2(x, *p):
    """
    Create a gaussian from amplitude, mean and variance

    :param x: [numpy array]    x axis data (array/list)
    :param p: [Tuple]         (a = amplitude, b = mean , c = variance)

    Returns a Gaussian array one value for each x value
    """
    a, b, c = p
    return a * np.exp(-0.5 * (x - b) ** 2 / (c ** 2))


def gaussian2d_variant1(x, y, *p):
    """
    Create a 2D gaussian from amplitude, mean x, variance x, mean y and
    variance y

    :param x: [numpy array]    x axis data (array/list)
    :param y: [numpy array]    y axis data (array/list)
    :param p: [Tuple]         (amplitude, mean x, variance x,
                               mean y, variance y)

    Returns a Gaussian array one value for each x value
    """
    A, x0, sigmax, y0, sigmay = p
    part1 = (x - x0) ** 2 / (2. * sigmax ** 2)
    part2 = (y - y0) ** 2 / (2. * sigmay ** 2)
    # return 2d gaussian
    return A * np.exp(-(part1 + part2))


def gaussian2d_variant2(x, y, *p, **kwargs):
    """
    Create a 2D gaussian from amplitude, mean x, variance x, mean y and
    variance y and a rotation angle theta (0.0 == x-axis) where theta is in
    degrees unless keyword units = radians or rad


    :param x: [numpy array]    x axis data (array/list)
    :param y: [numpy array]    y axis data (array/list)
    :param p: [Tuple]         (amplitude, mean x, variance x,
                               mean y, variance y, theta)
    :param kwargs:             keyword arguments i.e. units = "deg"
    keywords args are as follows:
        - units:               string either deg or rad

    Returns a Gaussian array one value for each x value
    """
    # extract values from p
    A, x0, sigmax, y0, sigmay, theta = p
    # sort out units
    units = kwargs.get('units', 'deg')
    if units == 'deg' or 'deg' in units:
        theta = np.deg2rad(theta)
    else:
        theta = theta
    # calculate gaussian
    a = np.cos(theta) ** 2 / (2. * sigmax ** 2) + np.sin(theta) ** 2 / (
    2. * sigmay ** 2)
    b = -np.sin(2 * theta) / (4. * sigmax ** 2) + np.sin(2 * theta) / (
    4. * sigmay ** 2)
    c = np.cos(theta) ** 2 / (2. * sigmay ** 2) + np.sin(theta) ** 2 / (
    2. * sigmax ** 2)
    part1 = a * (x - x0) ** 2
    part2 = -2 * b * (x - x0) * (y - y0)
    part3 = c * (y - y0) ** 2
    # return 2d gaussian
    return A * np.exp(-(part1 + part2 + part3))


def gaussian2d_variant3(x, y, *p):
    """
    Standard Gaussian function Creator
    :param x: [numpy array]    x axis data (array/list)
    :param y: [numpy array]    y axis data (array/list)
    :param p: [Tuple]         (amplitude, mean x, mean y, a, b, c)

    Returns a Gaussian array one value for each x value
    """
    A, x0, y0, a, b, c = p
    part1 = a * (x - x0) ** 2
    part2 = -2 * b * (x - x0) * (y - y0)
    part3 = c * (y - y0) ** 2
    # return 2d gaussian
    return A * np.exp(-(part1 + part2 + part3))


def sigma2percentile(sigma):
    """
    Percentile calculation from sigma
    i.e. 1 sigma == 0.68268949213708585 (68.27%)
    :param sigma: [float]     sigma value
    :return percentile:  [float] the percentile (i.e. between 0.00 and 1.00)
    """
    # percentile = integral of exp{-0.5x**2}
    percentile = erf(sigma / sqrt2)
    return percentile


def percentile2sigma(percentile):
    """
    Sigma calcualtion from percentile
    i.e. 0.68268949213708585 (68.27%) == 1 sigma
    :param percentile: [float]     percentile value (i.e. between 0.00 and 1.00)
    :return sigma:  [float] the sigma value (i.e. 1, 2, 3, 1.5)
    """
    # area = integral of exp{-0.5x**2}
    sigma = sqrt2 * erfinv(percentile)
    return sigma
