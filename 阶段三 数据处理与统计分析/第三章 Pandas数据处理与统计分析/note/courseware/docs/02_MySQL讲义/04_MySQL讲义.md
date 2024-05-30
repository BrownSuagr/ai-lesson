# 4 MySQL高级

## 11.三范式

### 11.1.什么是三范式

设计关系数据库时，遵从不同的规范要求，设计出合理的关系型数据库，这些不同的规范要求被称为不同的范式，各种范式呈递次规范，越高的范式数据库冗余越小。
### 11.2.数据冗余
数据冗余是指数据之间的重复，也可以说是同一数据存储在不同数据文件中的现象

![image-20220111094748806](img/image-20220111094748806.png)

![image-20220111094815762](img/image-20220111094815762.png)

### 11.3.范式的划分

根据数据库冗余的大小,目前关系型数据库有六种范式,各种范式呈递次规范，越高的范式数据库冗余越小。

六种范式：

- 第一范式（1NF）

- 第二范式（2NF）

- 第三范式（3NF）

- 巴斯-科德范式（BCNF）

- 第四范式 ( 4NF）

- 第五范式（5NF，又称完美范式）

一般遵循 前三种范式即可

### 11.4.一范式

第一范式（1NF）: 强调的是字段的原子性，即一个字段不能够再分成其他几个字段。

![image-20220111095035643](img/image-20220111095035643.png)

这种表结构设计就没有达到 1NF，要符合 1NF 我们只需把字段拆分，即：把 contact 字段拆分成 name 、tel、addr 等字段。

![image-20220111095043738](img/image-20220111095043738.png)

### 11.5.二范式

第二范式（2NF）: 满足 1NF的基础上，另外包含两部分内容

- 一是表必须有一个主键

- 二是非主键字段必须完全依赖于主键，而不能只依赖于主键的一部分

![image-20220111095145466](img/image-20220111095145466.png)

OrderDetail表的主键是什么？

主键的定义：能够确定唯一一行记录的特殊字段

![image-20220111095205270](img/image-20220111095205270.png)

```sql
create table test (
    name varchar(19),
    id int,
    primary key (name,id)
)
```

![image-20220111095248873](img/image-20220111095248873.png)

这里有主键由OrderID和ProductID共同组成

同时 UnitPrice 和 ProductName 这里两个字段 与ProductID的从属关系强于他们同OrderID的从属关系 

![image-20220111095302986](img/image-20220111095302986.png)

### 11.6 三范式

第三范式（3NF）: 满足 2NF

另外非主键字段必须直接依赖于主键，不能存在传递依赖。

即不能存在：非主键字段 A 依赖于非主键字段 B，非主键字段 B 依赖于主键的情况。

![image-20220111095424576](img/image-20220111095424576.png)

因为 OrderDate，CustomerID，CustomerName，CustomerAddr，CustomerCity 等非主键字段都完全依赖于主键（OrderID），所以符合 2NF。

不过问题是 CustomerName，CustomerAddr，CustomerCity 直接依赖的是 CustomerID（非主键列），而不是直接依赖于主键，它是通过传递才依赖于主键，所以不符合 3NF。

![image-20220111095444389](img/image-20220111095444389.png)

把【Order】表拆分为【Order】（OrderID，OrderDate，CustomerID）和【Customer】（CustomerID，CustomerName，CustomerAddr，CustomerCity）从而达到 3NF。

### 11.7.知识要点

范式:

设计关[数据库时，遵从不同的规范要求，设计出合理的关系型数据库，这些不同的规范要求被称为不同的范式，

各种范式呈递次规范，越高的范式数据库冗余越小。

三范式:

第一范式（1NF）: 强调的是列的原子性，即列不能够再分成其他几列。

第二范式（2NF）: 满足 1NF，另外包含两部分内容，一是表必须有一个主键；

二是非主键字段 必须完全依赖于主键，而不能只依赖于主键的一部分。

第三范式（3NF）: 满足 2NF，另外非主键列必须直接依赖于主键，不能存在传递依赖。

即不能存在：非主键列 A 依赖于非主键列 B，非主键列 B 依赖于主键的情况。





## 12.E-R模型及表间关系

### 12.1.E-R模型的使用场景

1. 对于大型公司开发项目，我们需要根据产品经理的设计，先使用建模工具, 如：power designer，db desinger等这些软件来画出实体-关系模型(E-R模型)

2. 然后根据三范式设计数据库表结构

### 12.2.E-R模型

E-R模型即实体-关系模型

E-R模型就是描述数据库存储数据的结构模型

![image-20220111095817454](img/image-20220111095817454.png)

### 12.3.三种关系

表现形式

- 实体: 用矩形表示，并标注实体名称

- 属性: 用椭圆表示，并标注属性名称

- 关系: 用菱形表示，并标注关系名称

E-R模型中的三种关系

- 一对一

  ![image-20220111095929110](img/image-20220111095929110.png)

- 一对多(1-n)

  ![image-20220111095943842](img/image-20220111095943842.png)

- 多对多(m-n)

  ![image-20220111095957194](img/image-20220111095957194.png)

  

### 12.4.知识要点

1. E-R模型由 实体、属性、实体之间的关系构成，主要用来描述数据库中表之间的关系和表结构。

2. 开发流程是先画出E-R模型，然后根据三范式设计数据库中的表结构



## 13.Python连接MySQL数据库

### 13.1.PyMysql模块介绍

如果使用之前学习的MySQL客户端来完成这个操作，那么这个工作量无疑是巨大的。我们可以通过使用程序代码的方式去连接MySQL数据库，然后对MySQL数据库进行增删改查的方式，实现10000条数据的插入，像这样使用代码的方式操作数据库就称为数据库编程。

安装pymysql第三方包：pip install pymysql

### 13.2.PyMysql模块使用

pymysql使用步骤

1. 导入pymysql 包

   ```python
   import pymysql
   ```

2. 创建连接对象

   ```python
   调用pymysql模块中的connect()函数来创建连接对象,代码如下:
   conn=connect(参数列表) 
       参数host：连接的mysql主机，如果本机是'localhost’ 
       参数port：连接的mysql主机的端口，默认是3306 
       参数user：连接的用户名
       参数password：连接的密码 
       参数database：数据库的名称 
       参数charset：通信采用的编码方式，推荐使用utf8
     连接对象conn的相关操作
       关闭连接 conn.close()
       提交数据 conn.commit()
       撤销数据 conn.rollback()
   
   ```

3. 获取游标对象

   ```python
   获取游标对象的目标就是要执行sql语句，完成对数据库的增、删、改、查操作。代码如下:
       调用连接对象的cursor()方法
       获取游标对象 cur =conn.cursor()
   游标操作说明:
       使用游标执行SQL语句: execute(operation [parameters ]) 执行SQL语句，返回受影响的行数，主要用于执行insert、update、delete、select等语句
       获取查询结果集中的一条数据:cur.fetchone()返回一个元组, 如 (1,'张三’)
       获取查询结果集中的所有数据: cur.fetchall()返回一个元组, 如 ((1,'张三'),(2,'李四’))
       关闭游标: cur.close(),表示和数据库操作完成
   ```

   

4. pymysql完成数据的增删改查操作

   ```python
   增删改查的sql语句
   sql = select * from 数据表
   
   执行sql语句完成相关数据操作
   游标cursor.execute(sql)
   
   ```

   

5. 关闭游标和连接

   ```python
   注意顺序
   
   先关闭游标
   cur.close()
   
   后关闭连接
   conn.close()
   
   ```

### 13.3.示例代码

```python
# 1 导入mysql连接库
import pymysql  

# 2 建立mysql连接
# 2.1 数据库信息
config = {'host': '192.168.88.161',  
          'user': 'root',  # 用户名
          'password': '123456',  # 密码
          'port': 3306,  # 端口，默认为3306
          'database': 'test',  # 数据库名称
          'charset': 'utf8'  # 字符编码
          }

# 2.2 连接到数据库
con = pymysql.connect(**config)

# 3 获得游标对象
cursor = con.cursor()

# 4 执行操作
# 4.1 查询表
cursor.execute("show tables")
# 4.2 读出所有库
table_list = [t[0] for t in cursor.fetchall()]
# 4.3 查找数据库是否存在目标表，如果没有则新建
table_name = 'score'  # 要写库的表名
if not table_name in table_list:  # 如果目标表没有创建 
    cursor.execute('''
    CREATE TABLE %s (
    userid           VARCHAR(20),
    score            int(2),
    group            VARCHAR(10),
    insert_date      VARCHAR(20)
    )ENGINE=InnoDB DEFAULT CHARSET=utf8
    ''' % table_name)  # 创建新表
    
    
# 4.4 写库
value0 = "20220101"
value1 = 100
value2 = "01"
timestamp = "20220331"

insert_sql = "INSERT INTO `%s` VALUES ('%s',%s,'%s','%s')" % \
             (table_name, value0, value1, value2, timestamp)  # 写库SQL依据
cursor.execute(insert_sql)  # 执行SQL语句，execute函数里面要用双引号
con.commit()  # 提交命令

# 5 关闭游标和连接
cursor.close()  # 关闭游标
con.close()  # 关闭数据库连接
```



### 13.4.知识要点

   pymysql使用步骤

   ①导入 pymysql 包

    import pymysql

   ②创建连接对象

    connect()

   ③获取游标对象

    连接对象.cursor()

   ④ pymysql完成数据的查询操作

    游标对象.execute()

   ⑤ 关闭游标和连接

    游标对象.close()   
    
    连接对象.close()


​      

   







