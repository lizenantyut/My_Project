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

###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    #expand参数会把切割出来的内容当成单独的一列 不加的话会报错
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    # [(-0.001, 7.91] < (7.91, 14.454] < (14.454, 31.0] <(31.0, 512.329]] 这个是按出现的频率均分。
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #将ndarray or Series 几等分 例如这里是4等分 看数据属于哪个桶
    # [(-0.08, 16.0] < (16.0, 32.0] < (32.0, 48.0] < (48.0, 64.0] <(64.0, 80.0]] 这个是按数值大小均分
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

#这里是对姓氏进行汇总 设定标准为10 如果船上的人的姓氏和小于10 那么统一称为Misc
stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
#print(data1['AgeBin'].value_counts())
print("-"*10)

#preview data again
print(data1.info())
print(data_val.info())
print(data1.sample(10))
#####################################################转换数据#######################################################

#code categorical data
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


#####################################################特征选择#######################################################

#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
#这里是把连续变量age,fare等连续型变量变为离散型变量
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


#define x and y variables for dummy features original
#one-hot encoding 特征值提取 方法是 get_dummies
#one-hot的基本思想：将离散型特征的每一种取值都看成一种状态，若你的这一特征中有N个不相同的取值，
# 那么我们就可以将该特征抽象成N种不同的状态，one-hot编码保证了每一个取值只会使得一种状态处于“激活态”，
# 也就是说这N种状态中只有一个状态位值为1，其他状态位都是0。
# 举个例子，假设我们以学历为例，我们想要研究的类别为小学、中学、大学、硕士、博士五种类别，
# 我们使用one-hot对其编码就会得到：

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')


#####################################切割训练级数据和测试数据 防止过拟合##################################
#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(
    data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(
    data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
    data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()


#####################################用统计学进行探索性的分析 终于到了画图时刻##################################
