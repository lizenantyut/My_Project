#coding: utf8
#matplotlib  seaborn 都可以用来画图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


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

#直方图
def zft_test():
    # 数据准备
    a = np.random.randn(100)
    s = pd.Series(a)
    # 用Matplotlib画直方图
    plt.hist(s)
    plt.show()
    # 用Seaborn画直方图
    sns.distplot(s, kde=False)
    plt.show()
    sns.distplot(s, kde=True)
    plt.show()

#条形图
def txt_test():
    # 数据准备
    x = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
    y = [5, 4, 8, 12, 7]
    # 用Matplotlib画条形图
    plt.bar(x, y)
    plt.show()
    # 用Seaborn画条形图
    sns.barplot(x, y)
    plt.show()

#箱线图
def xxt_test():
    # 数据准备
    # 生成10*4维度数据
    data = np.random.normal(size=(10, 4))
    lables = ['A', 'B', 'C', 'D']
    # 用Matplotlib画箱线图
    plt.boxplot(data, labels=lables)
    plt.show()
    # 用Seaborn画箱线图
    df = pd.DataFrame(data, columns=lables)
    sns.boxplot(data=df)
    plt.show()

#饼图
def bt_test():
    # 数据准备
    nums = [25, 37, 33, 37, 6]
    labels = ['High-school', 'Bachelor', 'Master', 'Ph.d', 'Others']
    # 用Matplotlib画饼图
    plt.pie(x=nums, labels=labels)
    plt.show()

#热力图
def rlt_test():
    # 数据准备
    df_flights = sns.load_dataset("flights",data_home=os.getcwd()+"\\"+'seaborn-data',cache=True,)
    data = df_flights.pivot('year', 'month', 'passengers')
    # print(data)
    # #用Seaborn画热力图
    sns.heatmap(data)
    plt.show()


# sdt_test()
# zxt_test()
# zft_test()
# txt_test()
# xxt_test()
# bt_test()
rlt_test()