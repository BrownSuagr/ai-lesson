# 1、导入pymysql模块
import pymysql

# 2、创建连接对象
conn = pymysql.connect(host='192.168.88.161', port=3306, user='root', password='123456', database='db_itheima', charset='utf8')

# 3、获取游标对象
cursor = conn.cursor()

# 4、定义SQL语句
sql = "select * from students"

# 5、执行SQL语句
row_count = cursor.execute(sql)
print(f'受影响的行数：{row_count}')

# 6、读取数据表中的数据
# print(cursor.fetchall())
for row in cursor.fetchall():
    print(row)

# 7、关闭游标对象与连接对象
cursor.close()
conn.close()
