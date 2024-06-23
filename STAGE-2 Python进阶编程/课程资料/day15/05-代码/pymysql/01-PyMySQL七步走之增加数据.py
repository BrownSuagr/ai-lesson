# 1、导入模块
import pymysql
# 2、创建连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')
# 3、获取游标
cursor = conn.cursor()
# 4、编写sql语句
sql = "insert into tb_student values (null, '关羽', 'male', 37, '13566007554');"
# 5、执行sql语句
row_count = cursor.execute(sql)
conn.commit()

if row_count > 0:
    print('数据添加成功！')
else:
    print('数据添加失败！')

# 6、关闭游标
cursor.close()
# 7、关闭连接
conn.close()