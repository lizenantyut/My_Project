#coding:utf8
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import warnings


def prepare_pack():
    """
    导入包的引用以及版本
    :return:
    """
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

    # load packages
    import sys  # access to system parameters https://docs.python.org/3/library/sys.html
    print("Python version: {}".format(sys.version))

    import \
        pandas as pd  # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
    print("pandas version: {}".format(pd.__version__))

    import matplotlib  # collection of functions for scientific and publication-ready visualization
    print("matplotlib version: {}".format(matplotlib.__version__))

    import numpy as np  # foundational package for scientific computing
    print("NumPy version: {}".format(np.__version__))

    import scipy as sp  # collection of functions for scientific computing and advance mathematics
    print("SciPy version: {}".format(sp.__version__))

    import IPython
    from IPython import display  # pretty printing of dataframes in Jupyter notebook
    print("IPython version: {}".format(IPython.__version__))

    import sklearn  # collection of machine learning algorithms
    print("scikit-learn version: {}".format(sklearn.__version__))

    # misc libraries
    import random
    import time

    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    print('-' * 25)
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    # from subprocess import check_output
    # print(check_output(["ls", "../input"]).decode("utf8"))

    # Any results you write to the current directory are saved as output.

#prepare_pack()