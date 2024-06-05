# 1、导入模块
import pymysql
# 2、创建连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='db_itheima', charset='utf8')
# 3、获取游标
cursor = conn.cursor()
# 4、编写sql语句
sql = "insert into tb_student values (null, '大乔', 'female', 19, '13566007566');insert into tb_student values (null, '小乔', 'female', 18, '18966007551');"
for i in sql.split(';')[0:-1]:
    # 5、执行sql语句
    row_count = cursor.execute(i)
    conn.commit()

print('数据批量添加成功！')

# 6、关闭游标
cursor.close()
# 7、关闭连接
conn.close()