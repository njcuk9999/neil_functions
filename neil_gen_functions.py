import numpy as np

# =============================================================================
# Define variables
# =============================================================================

# =============================================================================
# Define functions
# =============================================================================
def timestamp(types=None):
    """
    ============================================================================
    Creates a timestamp string
    ============================================================================
        mode 0: YYYY-MM-DD_HH-MM-SS    (Default)
        mode 1: HH:MM:SS 
        mode 2: YYYY/MM/DD
    """
    now = datetime.datetime.now()
    if(types==1):
        today = str(now.hour)+":"+str(now.minute)+":"+str(now.second)
    elif(types==2):
        today = str(now.year)+"/"+str(now.month)+"/"+str(now.day)
    else:
        today = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+\
                "_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
    return today
    

def makedirs(folder):
    """
    ==========================================================================
    Make dirs
    ==========================================================================
    Checks whether plot folder, subfolder and subsubfolder exist and
    if not creates the folders and folder path needed 
    (warning try to create full path, so folder needs to be correct)
    :param folder: 
    """
    temp = folder.split('/')
    xlink, xlink1 = '', ''
    made = False
    if len(temp[0]) == 0:
        temp[0] = '/'
    for j in ['..', '.', '/']:
        if temp[0] == j:
            for i in range(1, len(temp)):
                xlink = xlink + '/' + temp[i]
                if temp[i] not in os.listdir(j + xlink1) and temp[i] != '':
                    os.makedirs(j + xlink)
                    made = True
                    if prt:
                        print "makedirs"
                    tempdir = os.getcwd()
                    templink = folder.split('../')
                    if len(templink) == 1:
                        templink = folder.split('./')
                    print(str(timestamp(1)) + ": Made Folder: " + tempdir
                          + '/' + templink[-1])
                xlink1 = xlink
        if temp[0] not in ['..', '.', '/']:
            print ('\n' + str(timestamp(1)) +
                   ': Error: folder format not understood: ' + folder)
            made = True
    if not made:
        print('\n' + str(timestamp(1)) + ': All folders found for: ' +
              folder)
    return


def printcoeffs(p, f=2, xname='x', yname='f', formatx=None,
                errorlower=None, errorupper=None):
    """
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
    """
    # deal with formatting
    if formatx == 'latex':
        hat0, hat1 = '^{', '}'
        base0, base1 = '_{', '}'
        a0, a1 = '{hat0}', '{hat1}'
        b0, b1 = '{base0}', '{base1}'
        pm = r'\pm'
        ext = '$'
    else:
        hat0, hat1, hat2 = '^', '', ''
        base0, base1 = '', ''
        pm = '+-'
        a0, a1 = '+', ''
        b0, b1 = '-', ''
        ext = ''
    # deal with number of decimal places in coefficents
    d = '.{0}f'.format(f)
    # define length of coefficents (and assumed epu and epl)
    N = len(p)
    # deal formatting in the case of:
    #   1. no errors
    #   2. only error lower
    #   3. both error lower and uppwer
    if errorlower is None and errorupper is None:
        part0 = '{p:+' + d + '}'
        part1 = part0 + '{x}'
        partn = part1 + '{hat0}{n:.0f}{hat1}'

    elif errorupper is None:
        part0 = '[{p:+' + d + '}' + pm + '{epl:' + d + '}]'
        part1 = part0 + '{x}'
        partn = part1 + '{hat0}{n:.0f}{hat1}'
        epl = errorlower
        epu = np.zeros(N)
    else:
        part0 = ('[{p:+' + d + '}' + a0 + '+{epu:' + d + '}' + a1 +
                 b0 + '-{epl:' + d + '}' + b1 + ']')
        part1 = part0 + '{x}'
        partn = part1 + '{hat0}{n:.0f}{hat1}'
        epl = errorlower
        epu = errorupper
    # loop round p and construct coefficent string
    stringx = '{0}{1}({2})='.format(ext, yname, xname)
    for n in range(len(p)):
        # set up format dictionary
        fmt = dict(x=xname, p=p[n], epl=epl[n], epu=epu[n], n=N-n-1,
                   hat0=hat0, hat1=hat1, base0=base0, base1=base1)
        # if p[n] is zero skip
        if p[n] == 0:
            continue
        # if power=0 do not print +Ax^0, print +A
        if N - n -1 == 0:
            stringx += part0.format(**fmt)
        # if power=1 do not print +Ax^1, print Ax
        elif N - n -1 == 1:
            stringx += part1.format(**fmt)
        # else print +Ax^power
        else:
            stringx += partn.format(**fmt)
    stringx += ext
    # # get rid of any ++ and -+ and +-
    # stringx = stringx.replace('++', '+')
    # stringx = stringx.replace('-+', '-')
    # stringx = stringx.replace('+-', '-')
    # return string x
    return stringx
