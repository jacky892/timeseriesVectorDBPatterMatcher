import os
import logging
import pandas as pd
logging.basicConfig()
logger=logging.getLogger()
#logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.ERR)
logger.setLevel(logging.WARN)

def showtable(df, level=logging.DEBUG):
    '''show table if from jupyter, just print if not'''
    if isnotebook():
        display(df)
    else:
        if level==logging.DEBUG:
            logger.debug(df)

def isnotebook():
    ''' check if running from jupyter'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def log(msg):
    logger.info(msg)

def dlog(msg):
    logger.debug(msg)



class logUtil:

    def dlog(v, *argv):
        '''call dlog for each input argument'''
        logUtil.display_dlog(v)
        for v1 in argv:
            logUtil.display_dlog(v1)

    def display_dlog(msg, level=logging.DEBUG):
        '''
        if msg is  dataframe and in jupyter, use display, otherwise, use logger.debug
        '''
        if type(msg) in [type(pd.DataFrame(dtype='float64')), type(pd.Series(dtype='float64'))]:
            showtable(msg, level)
        else:
            logger.debug(msg)


