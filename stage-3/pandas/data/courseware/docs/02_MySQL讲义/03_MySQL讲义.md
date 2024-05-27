# 3 MySQL基础III

## 8.DQL数据查询语言

### 8.1.准备工作

```mysql
#创建商品表：
create table product(
 pid int primary key,
 pname varchar(20),
 price double,
 category_id varchar(32)
);
INSERT INTO product(pid,pname,price,category_id) VALUES(1,'联想',5000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(2,'海尔',3000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(3,'雷神',5000,'c001');

INSERT INTO product(pid,pname,price,category_id) VALUES(4,'杰克琼斯',800,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(5,'真维斯',200,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(6,'花花公子',440,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(7,'劲霸',2000,'c002');

INSERT INTO product(pid,pname,price,category_id) VALUES(8,'香奈儿',800,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(9,'相宜本草',200,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(10,'面霸',5,'c003');

INSERT INTO product(pid,pname,price,category_id) VALUES(11,'好想你枣',56,'c004');
INSERT INTO product(pid,pname,price,category_id) VALUES(12,'香飘飘奶茶',1,'c005');
INSERT INTO product(pid,pname,price,category_id) VALUES(13,'海澜之家',1,'c002');
```

### 8.2.语法：

```mysql
select [distinct]
*| 列名,列名
from 表
where 条件
```

### 8.3.简单查询

```mysql
#1.查询所有的商品.  
select *  from product;
#2.查询商品名和商品价格. 
select pname,price from product;
#3.别名查询.使用的关键字是as（as可以省略的）.  
#3.1表别名: 
select * from product as p;
#3.2列别名：
select pname as pn from product; 
#4.去掉重复值.  
select distinct price from product;
#5.查询结果是表达式（运算查询）：将所有商品的价格+10元进行显示.
select pname,price+10 from product;
```

### 8.4.条件查询

| 比较运算符 | > < <= >= = <> !=                | 大于、小于、大于(小于)等于、不等于                           |
| ---------- | -------------------------------- | ------------------------------------------------------------ |
|            | BETWEEN ... AND ...              | 显示在某一区间的值(含头含尾)                                 |
|            | IN(set)                          | 显示在in列表中的值，例：in(100,200)                          |
|            | LIKE '张%' Like '周_'LIKE '%涛%' | 模糊查询，Like语句中，%代表零个或多个任意字符，_代表一个字符，例如：first\_name like '\_a%'; |
|            | IS NULLIS NOT NULL               | 判断是否为空                                                 |
| 逻辑运算符 | and                              |                                                              |
|            | or                               | 多个条件任一成立                                             |
|            | not                              | 不成立，例：where not(salary>100);                           |

示例:

```mysql
#查询商品名称为“花花公子”的商品所有信息：
SELECT * FROM product WHERE pname = '花花公子';

#查询价格为800商品
SELECT * FROM product WHERE price = 800;

#查询价格不是800的所有商品
SELECT * FROM product WHERE price != 800;
SELECT * FROM product WHERE price <> 800;
SELECT * FROM product WHERE NOT(price = 800);

#查询商品价格大于60元的所有商品信息
SELECT * FROM product WHERE price > 60;


#查询商品价格在200到1000之间所有商品
SELECT * FROM product WHERE price >= 200 AND price <=1000;
SELECT * FROM product WHERE price BETWEEN 200 AND 1000;

#查询商品价格是200或800的所有商品
SELECT * FROM product WHERE price = 200 OR price = 800;
SELECT * FROM product WHERE price IN (200,800);

#查询含有'霸'字的所有商品
SELECT * FROM product WHERE pname LIKE '%霸%';

#查询以'香'开头的所有商品
SELECT * FROM product WHERE pname LIKE '香%';

#查询第二个字为'想'的所有商品
SELECT * FROM product WHERE pname LIKE '_想%';

#匹配四个字的所有商品 ____

#查询没有分类的商品
SELECT * FROM product WHERE category_id IS NULL;

#查询有分类的商品
SELECT * FROM product WHERE category_id IS NOT NULL;

```

### 8.5.排序查询

通过order by语句，可以将查询出的结果进行排序。暂时放置在select语句的最后。

格式:

```mysql
SELECT * FROM 表名 ORDER BY 排序字段 ASC|DESC;
 ASC 升序 (默认)
 DESC 降序

#1.使用价格排序(降序)
SELECT * FROM product ORDER BY price DESC;
#2.在价格排序(降序)的基础上，以分类排序(降序)
SELECT * FROM product ORDER BY price DESC,category_id DESC;
#3.显示商品的价格(去重复)，并排序(降序)
SELECT DISTINCT price FROM product ORDER BY price DESC;
```

### 8.6.聚合查询

之前我们做的查询都是横向查询，它们都是根据条件一行一行的进行判断，而使用聚合函数查询是纵向查询，它是对一列的值进行计算，然后返回一个单一的值；另外聚合函数会忽略空值。

今天我们学习如下五个聚合函数：

| 聚合函数 | 作用                                                         |
| -------- | ------------------------------------------------------------ |
| count()  | 统计指定列不为NULL的记录行数；                               |
| sum()    | 计算指定列的数值和，如果指定列类型不是数值类型，那么计算结果为0 |
| max()    | 计算指定列的最大值，如果指定列是字符串类型，那么使用字符串排序运算； |
| min()    | 计算指定列的最小值，如果指定列是字符串类型，那么使用字符串排序运算； |
| avg()    | 计算指定列的平均值，如果指定列类型不是数值类型，那么计算结果为0 |

示例:

```mysql
#1 查询商品的总条数
SELECT COUNT(*) FROM product;
#2 查询价格大于200商品的总条数
SELECT COUNT(*) FROM product WHERE price > 200;
#3 查询分类为'c001'的所有商品的总和
SELECT SUM(price) FROM product WHERE category_id = 'c001';
#4 查询分类为'c002'所有商品的平均价格
SELECT AVG(price) FROM product WHERE category_id = 'c002';
#5 查询商品的最大价格和最小价格
SELECT MAX(price),MIN(price) FROM product;
```

### 8.7.分组查询

分组查询是指使用group by字句对查询信息进行分组。

格式：

**SELECT** 字段1,字段2… **FROM** 表名 **GROUP** BY分组字段 **HAVING** 分组条件;

分组操作中的having子语句，是用于在分组后对数据进行过滤的，作用类似于where条件。

**having** 与**where**  的区别:

1).having是在分组后对数据进行过滤.,where是在分组前对数据进行过滤

2).having后面可以使用分组函数(统计函数),where后面不可以使用分组函数。

示例:

```mysql
#1 统计各个分类商品的个数
SELECT category_id ,COUNT(*) FROM product GROUP BY category_id ;
#2 统计各个分类商品的个数,且只显示个数大于1的信息
SELECT category_id ,COUNT(*) FROM product GROUP BY category_id HAVING COUNT(*) > 1;
```

### 8.8.分页查询

分页查询在项目开发中常见，由于数据量很大，显示屏长度有限，因此对数据需要采取分页显示方式。例如数据共有30条，每页显示5条，第一页显示1-5条，第二页显示6-10条。

- 格式：

```mysql
SELECT 字段1，字段2... FROM 表名 LIMIT M,N
M: 整数，表示从第几条索引开始，计算方式 （当前页-1）*每页显示条数
N: 整数，表示查询多少条数据 offset 偏移量（每页显示多少条数据）
SELECT 字段1，字段2... FROM 表名 LIMIT 0,5
SELECT 字段1，字段2... FROM 表名 LIMIT 5,5

#查询product表的前5条记录
SELECT *  FROM product LIMIT 0,5
等价于
SELECT *  FROM product LIMIT 5
```

### 8.9.insert into select语句

INSERT INTO SELECT 语句从一个表复制数据，然后把数据插入到一个已存在的表中。

基本语法:

```mysql
INSERT INTO table2
SELECT column_name(s)
FROM table1;
```

示例:

```mysql
create table product2(
 pid int primary key,
 pname varchar(20),
 price double
);

insert into product2 select pid,pname,price from product where category_id = 'c001';
```

![image-20211229180329061](img/image-20211229180329061.png)

## 9.多表操作

实际开发中，一个项目通常需要很多张表才能完成。例如：一个商城项目就需要分类表(category)、商品表(products)、订单表(orders)等多张表。且这些表的数据之间存在一定的关系，接下来我们将在单表的基础上，一起学习多表方面的知识。

![image-20211229180340531](img/image-20211229180340531.png)

### 9.1.表与表之间的关系

- 一对多关系：
  - 常见示例：客户和订单，分类和商品，部门和员工.
  - 一对多建表原则：在从表(多方)创建一个字段，字段作为外键指向主表(一方)的主键.

![image-20211229180352847](img/image-20211229180352847.png)

- 一对一关系：

一个学生对应一个学生档案材料，或者每个人都有唯一的身份证编号。

- 一对多关系：

一个学生只属于一个班，但是一个班级有多名学生。

- 多对多关系：

一个学生可以选择多门课，一门课也有多名学生。

### 9.2.外键约束

现在我们有两张表&quot;分类表&quot;和&quot;商品表&quot;，为了表明商品属于哪个分类，通常情况下，我们将在商品表上添加一列，用于存放分类cid的信息，此列称为：外键

![image-20211229180404921](img/image-20211229180404921.png)

![image-20211229180416325](img/image-20211229180416325.png)

此时&quot;分类表category&quot;称为：主表，&quot;cid&quot;我们称为主键。&quot;商品表products&quot;称为：从表，category_id称为外键。我们通过主表的主键和从表的外键来描述主外键关系，呈现就是一对多关系。

外键特点：
- 从表外键的值是对主表主键的引用。
- 从表外键类型，必须与主表主键类型一致。

**声明外键约束**

语法：

```mysql
alter table 从表 add [constraint] [外键名称] foreign key (从表外键字段名) references 主表 (主表的主键);
```

[外键名称] 用于删除外键约束的，一般建议&quot;_fk&quot;结尾
alter table 从表  drop foreign key 外键名称

- 使用外键目的：
  - 保证数据完整性

### 9.3.一对多操作

#### 9.3.1.分析

![image-20211229180430198](img/image-20211229180430198.png)

- category分类表，为一方，也就是主表，必须提供主键cid
- products商品表，为多方，也就是从表，必须提供外键category_id

#### 9.3.2.实现：分类和商品

```mysql
###创建分类表
create table category(
  cid varchar(32) PRIMARY KEY ,
  cname varchar(100)  #分类名称
);

# 商品表
CREATE TABLE products (
  pid varchar(32) PRIMARY KEY,
  name VARCHAR(40) ,
  price DOUBLE ,
  category_id varchar(32)
);

#添加约束
alter table products add constraint product_fk foreign key (category_id) references category (cid);

```

#### 9.3.3.操作

```mysql
#1 向分类表中添加数据
INSERT INTO category (cid ,cname) VALUES('c001','服装');

#2 向商品表添加普通数据,没有外键数据，默认为null
INSERT INTO products (pid,pname) VALUES('p001','商品名称');

#3 向商品表添加普通数据，含有外键信息(category表中存在这条数据)
INSERT INTO products (pid ,pname ,category_id) VALUES('p002','商品名称2','c001');

#4 向商品表添加普通数据，含有外键信息(category表中不存在这条数据) -- 失败,异常
INSERT INTO products (pid ,pname ,category_id) VALUES('p003','商品名称2','c999');

#5 删除指定分类(分类被商品使用) -- 执行异常
DELETE FROM category WHERE cid = 'c001';
```

## 10.多表查询

![image-20211229180449606](img/image-20211229180449606.png)

```mysql
CREATE TABLE category (
  cid VARCHAR(32) PRIMARY KEY ,
  cname VARCHAR(50)
);
CREATE TABLE products(
  pid VARCHAR(32) PRIMARY KEY ,
  pname VARCHAR(50),
  price INT,
  flag VARCHAR(2),    #是否上架标记为：1表示上架、0表示下架
  category_id VARCHAR(32),
  CONSTRAINT products_fk FOREIGN KEY (category_id) REFERENCES category (cid)
);
```



### 10.1.初始化数据

```mysql
#分类
INSERT INTO category(cid,cname) VALUES('c001','家电');
INSERT INTO category(cid,cname) VALUES('c002','服饰');
INSERT INTO category(cid,cname) VALUES('c003','化妆品');
#商品
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p001','联想',5000,'1','c001');
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p002','海尔',3000,'1','c001');
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p003','雷神',5000,'1','c001');

INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p004','JACK JONES',800,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p005','真维斯',200,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p006','花花公子',440,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p007','劲霸',2000,'1','c002');

INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p008','香奈儿',800,'1','c003');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p009','相宜本草',200,'1','c003');
```



### 10.2.多表查询

1. 交叉连接查询(基本不会使用-得到的是两个表的乘积) [了解]

   语法： **select** \* **from** A,B;
2. 内连接查询(使用的关键字 inner join -- inner可以省略)

  - 隐式内连接： **select** \* **from** A,B **where** 条件;
  - 显示内连接： **select** \* **from** A **inner join** B **on** 条件;
3. 外连接查询(使用的关键字 outer join -- outer可以省略)
  - 左外连接：left outer join
    - **select** \* **from** A **left outer join** B **on** 条件;
  - 右外连接：right outer join
    - **select** \* **from** A **right outer join** B **on** 条件;

```mysql
#1.查询哪些分类的商品已经上架
#隐式内连接
SELECT DISTINCT c.cname FROM category c , products p 
 WHERE c.cid = p.category_id AND p.flag = '1';

#显示内连接
SELECT DISTINCT c.cname FROM category c 
 INNER JOIN products p ON c.cid = p.category_id 
 WHERE p.flag = '1';

#2.查询所有分类商品的个数
#左外连接
INSERT INTO category(cid,cname) VALUES('c004','奢侈品');
SELECT cname,COUNT(category_id) FROM category c 
 LEFT OUTER JOIN products p 
  ON c.cid = p.category_id 
 GROUP BY cname;
```

下面通过一张图说明连接的区别:

![image-20211229180506868](img/image-20211229180506868.png)

### 10.3.子查询

子查询：一条select语句结果作为另一条select语法一部分（查询条件，查询结果，表等）。

select ....查询字段 ... from ... 表.. where ... 查询条件

```mysql
#3 子查询, 查询“化妆品”分类上架商品详情
#隐式内连接
SELECT p.* FROM products p , category c 
 WHERE p.category_id=c.cid AND c.cname = '化妆品';

#子查询
##作为查询条件
SELECT * FROM products p 
 WHERE p.category_id = ( 
  SELECT c.cid FROM category c 
   WHERE c.cname='化妆品'
 );
 ##作为另一张表
 SELECT * FROM products p , 
   (SELECT * FROM category WHERE cname='化妆品') c 
  WHERE p.category_id = c.cid;

#查询“化妆品”和“家电”两个分类上架商品详情
SELECT * FROM products p 
 WHERE p.category_id in ( 
  SELECT c.cid FROM category c 
   WHERE c.cname='化妆品' or c.name='家电'
 );

```