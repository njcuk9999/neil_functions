import numpy as np

# =============================================================================
# Define variables
# =============================================================================

# =============================================================================
# Define functions
# =============================================================================
def Mj_from_spt(x):
    """
    Lepine Mj to SpT relationship (Lepine et al. 2013, Equation 23, page 24)
    :param x: numpy array, SpT
    :return:
    """
    return 5.680 + 0.393*x + 0.040*x**2


def convert_numspt_to_stringspt(x):
  """
  Convert a numerical spectral type to a string spectral type (i.e. 0.0 --> M0.0)
  :param x: numpy array of floats, numerical spectral types
  :return: numpy array of strings, string spectral type
  """
    stringspt = []
    spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']
    lowerlimits = [-np.inf, -48.0, -38.0, -28.0, -18.0, -8.0, 0.0,
                   10.0, 20.0, 30.0]
    upperlimits = [-48.0, -38.0, -28.0, -18.0, -8.0, 0.0,
                   10.0, 20.0, 30.0, np.inf]
    add_on = [58, 48, 38, 28, 18, 8, 0, -10, -20, -30]
    for i in range(len(x)):
        for j in range(len(spectral_types)):
            if lowerlimits[j] <= x[i] < upperlimits[j]:
                args = [spectral_types[j], x[i] + add_on[j]]
                stringspt.append('{0}{1}'.format(*args))
    return np.array(stringspt)
    
    
def convert_stringspt_to_numspt(x):
  """
  Convert a string spectral type to a numerical spectral type (i.e. M0.0 --> 0.0)
  :param x: numpy array of strings, string spectral type 
  :return: numpy array of floats, numerical spectral types
  """
    stringspt = []
    spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']
    lowerlimits = [-np.inf, -48.0, -38.0, -28.0, -18.0, -8.0, 0.0,
                   10.0, 20.0, 30.0]
    upperlimits = [-48.0, -38.0, -28.0, -18.0, -8.0, 0.0,
                   10.0, 20.0, 30.0, np.inf]
    add_on = [58, 48, 38, 28, 18, 8, 0, -10, -20, -30]
    for i in range(len(x)):
        for j in range(len(spectral_types)):
            if lowerlimits[j] <= x[i] < upperlimits[j]:
                args = [spectral_types[j], x[i] + add_on[j]]
                stringspt.append('{0}{1}'.format(*args))
    return np.array(stringspt)
    
