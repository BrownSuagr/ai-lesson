# 导入pymysql包
import pymysql
import sys


# 主函数入口
def main():

    host = '124.70.78.122'
    user = 'root'
    password = '!@#123qwe'
    database = 'bs'

    # 1、创建连接对象
    try:
        conn = pymysql.connect(host=host, port=3306, user=user, password=password, database=database)
    except ConnectionError as e:
        print(f'数据库连接错误：{e}')
        sys.exit()

    # 2、创建游标对象
    cursor = conn.cursor()

    # 3、定义SQL语句
    sql = 'select * from amm_user'

    # 4、执行查询语句
    try:
        cursor.execute(sql)
    except Exception as e:
        print(f'数据库SQL执行错误：{e}')

    # 5、获取全部结果集
    result = cursor.fetchall()
    print(f'result:{result}')
    for row in result:
        print(row)

    # 6、关闭游标和连接
    cursor.close()
    conn.close()


# 主函数执行入口
if __name__ == '__main__':
    main()