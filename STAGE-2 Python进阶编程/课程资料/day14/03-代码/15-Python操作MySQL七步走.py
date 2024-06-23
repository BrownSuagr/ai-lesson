# 1、导入模块
import pymysql

# 2、创建数据库链接
# host服务器的IP地址
# port端口号
# user用户名
# password密码
# database数据库名称
# charset编码格式
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')

# 3、获取游标（数据库操作同时通过游标实现）, 以后数据库的增删改查都要通过游标才能实现
cursor = conn.cursor()

# 4、定义SQL语句
sql = "insert into tb_student values (null, '关羽', 'male', 36, '18865007991');"

# 5、执行SQL语句
cursor.execute(sql)  # 执行SQL语句，返回受影响的行数（在内存中预演SQL执行，未真正提交到硬盘）
conn.commit()  # 提交事务（把刚才SQL的执行结果，写入到硬盘，让SQL语句立即生效）
print('数据添加成功！')

# 6、关闭游标
cursor.close()

# 7、关闭数据库连接
conn.close()