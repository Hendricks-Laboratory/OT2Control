'''
There are some nice functions that are useful for debugging and for multiple files, so I put them
here. Note that not all of these packages are needed, I just don't know which ones aren't needed
'''
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
import socket
import json
import dill
import math
import os
import shutil
import webbrowser
from tempfile import NamedTemporaryFile
import logging
import asyncio
import threading
import time
import subprocess
import functools

from bidict import bidict
import gspread
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import opentrons.execute
from opentrons import protocol_api, simulate, types
from boltons.socketutils import BufferedSocket

from Armchair.armchair import Armchair
import Armchair.armchair as armchair

#VISUALIZATION
def df_popout(df):
    '''
    Neat trick for viewing df as html in browser
    With some minor tweaks
    Credit to Stack Overflow
    @author Shovalt
    '''
    with NamedTemporaryFile(delete=False, suffix='.html') as f:
        html_str = df.to_html()
        f.write(str.encode(html_str,'ascii'))
    webbrowser.open(f.name)

def wslpath(path, flag):
    '''
    makes a wslpath call to system. path and flag are the same as for wslpath
    system call courtesy of stack overflow @author illkachu
    '''
    exec_str = "wslpath -{} '{}'".format(flag, path)
    x = subprocess.check_output(exec_str,shell=True).decode('utf8')[:-1]#drop newline
    return x

def make_unique(s):
    '''
    makes every element in s unique by adding _1, _2, ...
    params:
        pd.Series s: a series of elements with some duplicates
    returns:
        pd.Series: s with any duplicates made unique by adding a count
    e.g.
    ['sloth', 'gorilla', 'sloth'] becomes ['sloth_1', 'gorilla', 'sloth_2']
    '''
    val_counts = s.value_counts()
    duplicates = val_counts.loc[val_counts > 1]
    def _get_new_name(name):
        #mini helper func for the apply
        if name in duplicates:
            i = duplicates[name]
            duplicates[name] -= 1
            return "{}_{}".format(name, i)
        else:
            return name
    return s.apply(_get_new_name)
    webbrowser.open(f.name)


def error_exit(func):
    '''
    This is a decorator to wrap functions in error catching code. Methods with this
    decorator call error_handler upon failure. It is used for class methods
    Preconditions:  
        To use this wrapper, the client class must implement a function, _error_handler,
    '''
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            args[0]._error_handler(e)
    return decorated
