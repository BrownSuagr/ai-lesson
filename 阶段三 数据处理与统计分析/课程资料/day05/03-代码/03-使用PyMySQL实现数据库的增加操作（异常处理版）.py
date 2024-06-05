# 1、导入pymysql模块
import pymysql

# 2、创建连接对象
conn = pymysql.connect(host='192.168.88.161', port=3306, user='root', password='123456', database='db_itheima', charset='utf8')

# 3、获取游标对象
cursor = conn.cursor()

try:
    # 4、定义SQL语句
    sql = "insert into students values (null, '赵云', 22, 'male', 98.0, 3)"

    # 5、执行SQL语句
    row_count = cursor.execute(sql)
    print(f'受影响的行数：{row_count}')
    conn.commit()
except Exception as e:
    conn.rollback()

# 6、关闭游标对象
cursor.close()

# 7、关闭连接对象
conn.close()