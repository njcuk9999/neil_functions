# neil_functions
Program containing a set of custom functions I regularly use in Python


# Description of functions:

# General functions:

import neil_gen_functions.py

# timestamp(types=None):

          Creates a timestamp string
        
          :param types: integer, 0 1 or 2 see below:
        
                mode 0: YYYY-MM-DD_HH-MM-SS    (Default)
                mode 1: HH:MM:SS 
                mode 2: YYYY/MM/DD
                
          :return today: string, timestamp in format above



# makedirs(folder):
          Checks whether plot folder, subfolder and subsubfolder exist and
          if not creates the folders and folder path needed 
          (warning try to create full path, so folder needs to be correct)
          
          :param folder: string, location of folder to create



# printcoeffs(p, f=2, xname='x', yname='f', formatx=None, errorlower=None, errorupper=None):
    prints a nice version of coefficients
    
    :param p: list of floats, coefficents as in numpy.polyval
    :param f: integer, number of decimal places for coefficents
    :param xname: string, variable name assigned to x
    :param yname: string, variable name assigned to y
    :param formatx: None or string, format (currently None or "latex") if latex
                    use dollar notation to write string
    :param errorlower: None or list of floats, if None no uncertainties, if
                       error upper is None and error lower is not None assumes
                       uncertainties are equal i.e. +/-
                       error lower list must be same length as p
    :param errorupper: None or list of floats, error upper is None and
                       error lower is not None assumes uncertainties are equal
                       i.e. +/-
                       error upper list must be same length as p
