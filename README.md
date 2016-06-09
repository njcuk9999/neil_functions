# neil_functions
Program containing a set of custom functions I regularly use in Python (Generalised for the use of everyone)

# Table of contents

* [General functions](#general-functions)
  * [Timestamp function](#timestamp-function)
  * [Display percentage function](#display-percentage-function)
  * [Make directories function](#make-directories-function)
  * [Print coefficents function](#print-coefficents-function)
* [Math functions](#math-functions)
  * [One dimensional spline interpolator](#one-dimensional-spline-interpolator)
  * [Reduced chi-squared function](#reduced-chi-squared-function)
  * [Polyval with propagated uncertainties](#polyval-with-propagated-uncertainties)
* [Stats functions](#stats-functions)
* [Astro functions](#astro-functions)
  * [Absolute J band from spectral type](#absolute-J-band-from-spectral-type)
  * [Convert numerical spectral type to string spectral type](#convert-numerical-spectral-type-to-string-spectral-type)


## Description of functions:

### General functions:

```python
import neil_gen_functions
```

#### Timestamp function
```python
timestamp(types=None):
```
Creates a timestamp string

:param types: integer, 0 1 or 2 see below:

      mode 0: YYYY-MM-DD_HH-MM-SS    (Default)
      mode 1: HH:MM:SS 
      mode 2: YYYY/MM/DD
      
:return today: string, timestamp in format above

#### Display percentage function
```python
percentage(it1, total, message, ptype=None):
```
   Displays a simple message followed by a updating percentage
    bar, for use inside a loop, variables are as follows:

         - Format:
            percentage(it1, total, message, ptype)

      it1 (INT) is the iteration number of the loops

      total (INT) is the total number of iterations of the loop

      message (STRING) is displayed as follows:
          "[message] ...0%"
          "[message] ...50%"
          "[message] ...100%"

      ptype (STRING) is the format in which to return the percentage.
          Current accepted formats are:

          'i'         - returns percentage in integer form

              message ...12%

          'f0'         - returns percentage in integer form

              message ...12%

          'f2'         - returns percentage to two decimal places

              message ...12.34%

          'f4'         - returns percentage to four decimal places

              message ...12.3456%

          'bar'      - returns a loading percentage bar:

           Loading =================================================

#### Make directories function
```python
makedirs(folder)
```
   Checks whether plot folder, subfolder and subsubfolder exist and
   if not creates the folders and folder path needed 
   (warning try to create full path, so folder needs to be correct)
   
   :param folder: string, location of folder to create

#### Print coefficents function 
```python
printcoeffs(p, f=2, xname='x', yname='f', formatx=None, errorlower=None, errorupper=None)
```
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
                       
### Math functions

```python
import neil_math_functions
```

#### One dimensional spline interpolator 
```python
interp1d(x, y)
```
    This is a easier to use spline interpolation
    call by using F = interp1d(oldx, oldy)
    use by using newy = F(oldy)
    :param x: array of floats, must be in numerical order for spline to function
    :param y: array of floats, y values such that y = f(x) to be mapped onto new x values (with a cubic spline)
    :return:

#### Reduced chi-squared function
```python
rchisquared(x, y, ey, model, p)
```
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
 
#### Polyval with propagated uncertainties 
```python
polyval(p, x, ex)
```
    Numpy polyval command with uncertainties propagated using:
    
        y = sum { p_n * x^(N-n) }
        ey = sum { ((dy/dx_n)**2 + ex_n**2
        
    i.e. currently assumes no uncertainties in p
    
    :param p: array/list of floats, coefficient list (as in numpy.polyval)
    :param x: array of floats, x values such that y = f(x) 
    :param ex: array of floats, x uncertainties
    :return: y and ey (propagated uncertainties in y)
    
### Stats functions

```python
import stats_functions
```

#### Gaussian variants

```python
gaussian1d_variant1(x, *p):
```

    Create a normalised gaussian (Area == 1) from mean and variance

    :param x: [numpy array]    x axis data (array/list)
    :param p: [Tuple]         (mean , variance)

    Area of gaussian is 1  --> A = 1/(c*sqrt(2pi))

    Returns a Gaussian array one value for each x value

```python
gaussian1d_variant2(x, *p):
```

    Create a gaussian from amplitude, mean and variance

    :param x: [numpy array]    x axis data (array/list)
    :param p: [Tuple]         (a = amplitude, b = mean , c = variance)

    Returns a Gaussian array one value for each x value

#### 2D Gaussian variants

```python
gaussian2d_variant1(x, y, *p):
```

    Create a 2D gaussian from amplitude, mean x, variance x, mean y and
    variance y

    :param x: [numpy array]    x axis data (array/list)
    :param y: [numpy array]    y axis data (array/list)
    :param p: [Tuple]         (amplitude, mean x, variance x,
                               mean y, variance y)

    Returns a Gaussian array one value for each x value

```python
gaussian2d_variant2(x, y, *p, **kwargs):
```

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

```python
gaussian2d_variant3(x, y, *p):
```

    Standard Gaussian function Creator
    :param x: [numpy array]    x axis data (array/list)
    :param y: [numpy array]    y axis data (array/list)
    :param p: [Tuple]         (amplitude, mean x, mean y, a, b, c)

    Returns a Gaussian array one value for each x value

#### sigma to from percentile functions

```python
sigma2percentile(sigma):
```

    Percentile calculation from sigma
    i.e. 1 sigma == 0.68268949213708585 (68.27%)
    :param sigma: [float]     sigma value
    :return percentile:  [float] the percentile (i.e. between 0.00 and 1.00)


```python
percentile2sigma(percentile):
```

    Sigma calcualtion from percentile
    i.e. 0.68268949213708585 (68.27%) == 1 sigma
    :param percentile: [float]     percentile value (i.e. between 0.00 and 1.00)
    :return sigma:  [float] the sigma value (i.e. 1, 2, 3, 1.5)
    

### Astro functions

```python
import neil_astro_functions
```

#### Absolute J band from spectral type 
```python
Mj_from_spt(x)
```
    Lepine Mj to SpT relationship (Lepine et al. 2013, Equation 23, page 24)
    :param x: numpy array, SpT
    :return:

#### Convert numerical spectral type to string spectral type
```python
convert_numspt_to_stringspt(x)
```
    Convert a numerical spectral type to a string spectral type
    i.e.
         00.0 --> M0.0
         18.0 --> L8.0
        -01.0 --> K7.0
        -18.0 --> G0.0
    :param x: numpy array of floats, numerical spectral types
    :return: numpy array of strings, string spectral type
