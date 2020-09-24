#coding: utf8
#matplotlib  seaborn 都可以用来画图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#散点图
def sdt_test():
    """
    散点图测试
    :return:
    """
    # 数据准备
    N = 1000
    x = np.random.randn(N)
    y = np.random.randn(N)
    # 用Matplotlib画散点图
    plt.scatter(x, y,marker='x')
    plt.show()
    # 用Seaborn画散点图
    df = pd.DataFrame({'x': x, 'y': y})
    sns.jointplot(x="x", y="y", data=df, kind='scatter');
    plt.show()

#折线图
def zxt_test():
    # 数据准备
    x = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    y = [5, 3, 6, 20, 17, 16, 19, 30, 32, 35]
    # 使用Matplotlib画折线图
    plt.plot(x, y)
    plt.show()
    # 使用Seaborn画折线图
    df = pd.DataFrame({'x': x, 'y': y})
    sns.lineplot(x="x", y="y", data=df)
    plt.show()

#sdt_test()
zxt_test()