#coding:utf8
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import warnings
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix  #pandas 0.19之后用这个
#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

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
    from subprocess import check_output

def greet_data_1():
    """
    meet and greetdata
    """
    #pclass     船票等级 1 最高 2 次之 3 最低
    #Embarked   出发地

    pass

#prepare_pack()
#####################################################加载数据#######################################################
data_raw = pd.read_csv('train.csv')
data_val  = pd.read_csv("test.csv")
data1 = data_raw.copy(deep = True)
data_cleaner = [data1, data_val]
print (data_raw.info())
print(data_raw.sample(10))


print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)
print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)
data_raw.describe(include = 'all')


#####################################################清洗数据#######################################################
for dataset in data_cleaner:
    #complete missing age with median 用中位数填充
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode      用众数填充
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median 用中位数填充
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())