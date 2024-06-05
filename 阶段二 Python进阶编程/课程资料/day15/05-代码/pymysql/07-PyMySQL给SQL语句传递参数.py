# 1、导入模块
import pymysql

# 2、创建数据库连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')

# 3、获取游标
cursor = conn.cursor()

# 4、编写sql语句
name = input('请输入要添加学员的姓名：')
gender = input('请输入要添加学员的性别(male/female)：')
age = int(input('请输入要添加学员的年龄：'))
mobile = input('请输入要添加学员的电话：')

sql = "insert into tb_student values (null, %s, %s, %s, %s);"

# 5、执行sql语句
params = [name, gender, age, mobile]
row_count = cursor.execute(sql, params)  # SQL过滤，防止恶意SQL语句
conn.commit()

if row_count > 0:
    print('添加成功')
else:
    print('添加失败')
# 6、关闭游标
cursor.close()
# 7、关闭连接
conn.close()