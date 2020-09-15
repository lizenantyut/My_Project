#coding: utf8

def py_orcle():
    import cx_Oracle
    conn = cx_Oracle.connect('scott/wang@10.55.200.132/zdcomdb')
    cur = conn.cursor()
    sql = 'SELECT sysdate FROM dual'
    cur.execute(sql)
    data = cur.fetchmany()
    print(data)

    cur.close()
    conn.close()

    #db = cx_Oracle.connect(ora_dic['value'])

def py_mysql():
    import pymysql
    # 第一步：连接到数据库
    con = pymysql.connect(host="test.lemonban.com",  # 数据库的地址
                          user='xxxxx',  # 登录数据库的账号
                          password="xxxxx",  # 登录数据库的密码
                          port=3306,  # 端口
                          database='xxxxx',  # 库名称
                          )
    # 第二步：创建游标
    cur = con.cursor()
    # 第三步：执行对应的sql语句  方法：execute（）
    sql = 'SELECT * FROM students;'
    cur.execute(sql)

def py_sqlser():
    import pymssql
    # 第一步：连接到数据库
    con = pymssql.connect(host='xxx',  # 数据库的地址
                          user='xxx',  # 登录数据库的账号
                          password='xxxx',  # 登录数据库的密码
                          database='xxx')  # 库名称

    # 第二步：创建游标
    cur = con.cursor()
    # 第三步：执行对应的sql语句  方法：execute（）
    sql = 'SELECT * FROM students;'
    cur.execute(sql)

def py_mongodb():
    import pymongo
    # 第一步：建立连接
    client = pymongo.MongoClient("localhost", 27017)
    # 第二步：选取数据库
    db = client.test1
    # 第三步：选取集合
    stu = db.stu

    # 第四步：执行相关操作

    # 添加一条数据
    data1 = {1: 'musen', 2: 18}
    stu.insert_one(data1)
    # 获取一条数据
    s2 = stu.find_one()

def py_elasticsearch():
    #TODO
    pass

def py_redis():
    import redis
    st = redis.StrictRedis(
        host='localhost',  # 服务器本机
        port='6379',  # 端口：
        db=0,  # 库：
    )
    # redis操作的命令，对应st对象的方法
    # 比如在数据库中创建一条键为test的数据，往里面添加3个元素
    st.lpush('test', 11, 22, 33)
#py_orcle()
