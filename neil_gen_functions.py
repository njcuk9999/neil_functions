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
