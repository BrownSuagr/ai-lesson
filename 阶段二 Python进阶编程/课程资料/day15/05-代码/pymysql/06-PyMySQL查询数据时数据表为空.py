# 1、导入模块
import pymysql
# 2、创建连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')
# 3、获取游标
cursor = conn.cursor()
# 4、编写sql语句
sql = "select * from tb_student;"
# 5、执行sql语句
cursor.execute(sql)
# 取出数据（一条）
# data = cursor.fetchone()
# print(data)
# 取出数据（多条）
data = cursor.fetchall()
# print(data)
if data:
    for i in data:
        print(f'编号：{i[0]}')
        print(f'姓名：{i[1]}')
        print(f'性别：{i[2]}')
        print(f'年龄：{i[3]}')
        print(f'电话：{i[4]}')
        print('-' * 40)
else:
    print('暂未查询到任何记录!')
# 6、关闭游标
cursor.close()
# 7、关闭连接
conn.close()