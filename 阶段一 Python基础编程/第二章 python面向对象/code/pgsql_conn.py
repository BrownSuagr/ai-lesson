import sys
import psycopg2

# 创建连接对象
host = 'localhost'
user = 'root'
password = 'localhost'
database = 'database'


# 配置数据库连接参数
conn_params = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost"
}

# 建立连接
conn = psycopg2.connect(**conn_params)

# 创建一个游标对象
cur = conn.cursor()

# 执行一个查询
cur.execute("SELECT * FROM your_table LIMIT 5;")

# 获取查询结果
rows = cur.fetchall()
for row in rows:
    print(row)

# 关闭游标和连接
cur.close()
conn.close()
