import sys
import pymysql
import json


def main():
    # 创建连接对象
    host = '124.70.78.122'
    user = 'root'
    password = '!@#123qwe'
    database = 'bs'

    try:
        conn = pymysql.connect(host=host, port=3306, user=user, password=password, database=database)
    except ConnectionError as e:
        print(f'数据库连接错误：{e}')
        sys.exit()

    # 创建游标对象
    cursor = conn.cursor()
    # 执行查询语句
    sql = 'SELECT * FROM amm_user'

    try:
        cursor.execute(sql)
    except Exception as e:
        print(f'数据库SQL执行错误：{e}')

    result = cursor.fetchall()  # 获取全部结果集
    print(f'result:{result}')
    for row in result:
        print(row)

    # 关闭游标和连接
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
