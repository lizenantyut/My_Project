#coding: utf8
import pandas as pd
import numpy as np

#  一 加载数据
df=pd.read_csv("线路表2.csv",encoding='GBK')
#  二 切片切块 也可以叫选取数据
# ['CKXLID', 'XLMC', 'CKJGID', 'SZMC', 'MZMC', 'SZSCSJ', 'SZMCSJ','MZSCSJ', 'MZMCSJ', 'SXZDZH', 'XXZLC',
#  'XGSJ', 'COMPANYNM', 'FLAG','ACTION_FLAG']
df_test = df[['CKXLID','XLMC','XGSJ']]      #选取ckxlid xlmc xgsj这3列
df_test1 = df.iloc[0:5, 0:3]                #通过位置选择 前5行 前3列
df_test2 = df.iloc[[0, 3, 5], [0, 3]]       #通过位置选择 第1 4 6 行，第1，4列 从0开始 所以要加1
df_test3 = df.loc[0:5, 'CKXLID':'CKJGID']   #通过标签选择 前5行 前3列 注意这里列标签是0,1,2 如果比如是字符串 要换成字符串
df_test4 = df.loc[[0, 3, 5], ['CKXLID', 'CKJGID']]       #通过标签选择
df_test5 = df[df['CKXLID'].isin(['188905791','188905794'])]  #选取CKXLID列中值为'188905791','188905794'的行


#  三 过滤清洗
df_test = df.drop_duplicates('CKJGID', keep='first')   #去重重复行 first默认保持第一个。last后一个 false 都删除
df_test2 = df[(df['CKXLID'] >= 188905794) &  (df['CKXLID'] <= 188905796)] #选取CKXLID >=188905794 <=188905796

#  四 处理缺失值
df_test = df.dropna(how='any')                     #删除任何缺失数据的行
df_test = df.dropna(subset=['CKJGID'],how='any')   #删除某列为空的行
df_test2 = df.fillna(value='null')                 #空值用什么替代

#  五 统计
#df_test = df.groupby()

#  六 排序 sort
#df = df.sort_values(by=['col1','col2'], ascending=True)  #ascending=False 是降序 True是升序

#  七 创造新列从已存在的列

#  八 时间
datetime = pd.date_range(start='2020-01-01',end='2020-01-03')
print(datetime)