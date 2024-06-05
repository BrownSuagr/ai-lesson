# 1、导入模块
import pymysql
# 2、创建数据库连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')
# 3、获取游标
cursor = conn.cursor()
# 4、编写SQL语句
sql = "delete from tb_student where id = 2;"
# 5、执行SQL语句
cursor.execute(sql)
conn.commit()
print('删除数据成功！')
# 6、关闭游标
cursor.close()
# 7、关闭数据库连接
conn.close()