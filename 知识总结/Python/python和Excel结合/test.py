# coding: utf8
import json
import requests
import pandas as pd
import datetime
from openpyxl import load_workbook

data = []  # 用于存储每一行的Json数据
rq = datetime.datetime
sheet_date = str(datetime.date.today()+datetime.timedelta(days=-1))[:10]
title_date= str(datetime.date.today()+datetime.timedelta(days=-1))[5:10]
request_param = {"date": sheet_date}

def excelAddSheet(dataframe, excelWriter, sheet_name):
    book = load_workbook(excelWriter.path)
    excelWriter.book = book
    dataframe.to_excel(excel_writer=excelWriter, sheet_name=sheet_name, index=False)
    excelWriter.close()

def firstDayOfMonth(dt):
    """判断今天是不是这个月第一天"""
    now_day = (dt + datetime.timedelta(days=-dt.day + 1)).day
    return now_day == dt.day

def sub_date(x):
    if x is None:
        return None
    else:
        return x[-8:]

def request_post(url, param):
    fails = 0
    while True:
        try:
            if fails >= 20:
                break

            headers = {'content-type': 'application/json'}
            ret = requests.post(url, json=param, headers=headers, timeout=10)
            if ret.status_code == 200:
                text = json.loads(ret.text)
            else:
                continue
        except:
            fails += 1
            print('网络连接出现问题, 正在尝试再次请求: ', fails)
        else:
            break
    return text

def start():
    post_url = "http://10.55.202.164:40701/epidemic/getAbnormalOnBoardWarnBusDataList"

    a = request_post(post_url, request_param)
    js = json.dumps(a, sort_keys=True, indent=4, separators=(',', ':'))  # 格式化转json格式

    df = pd.read_json(js, orient='records')  # 如果文本里有汉字格式，此处需要设置encoding= 'UTF-8'，否则汉字会乱码
    print(df.columns)
    #改变列名
    df.rename(columns={'busName':'车辆名称', 'date':'日期', 'endCrowd':'结束时满载率', 'endOnBoard':'结束时在车人数', 'endTime':'结束时间', 'lineName':'线路',
           'plans':'额定人数', 'startCrowd':'开始时满载率', 'startOnBoard':'开始时在车人数',
           'startTime':'开始时间'}, inplace = True)
    #更改顺序
    df = df[['日期','车辆名称','线路','额定人数','开始时间','开始时在车人数','开始时满载率','结束时间','结束时在车人数','结束时满载率']]


    df['日期']=df['日期'].apply(lambda x: datetime.datetime.strftime(x,"%Y/%m/%d"))
    df['结束时间']=df['结束时间'].apply(lambda x: sub_date(x))
    df['开始时间']=df['开始时间'].apply(lambda x: sub_date(x))
    df = df.sort_values(by='开始时间', ascending=True)

    #判断是否是月初
    now = datetime.date.today()
    rq_flag = firstDayOfMonth(now)
    excelname = 'F:/py/集团实时客流满载率大于90%车辆明细-{}月份.xlsx'.format(datetime.datetime.now().month)
    if rq_flag:
        df.to_excel(excelname,sheet_name=title_date,index=False)
    else:
        excelWriter = pd.ExcelWriter(excelname,engine='openpyxl')
        excelAddSheet(df, excelWriter, title_date)

start()
