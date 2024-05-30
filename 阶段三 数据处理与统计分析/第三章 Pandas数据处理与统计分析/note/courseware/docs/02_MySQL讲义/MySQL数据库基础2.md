# 2 MySQL数据库基础II



## 九、SQL约束

回顾建表的基本语法：

```sql
create table 数据表名称(
	字段名称1 字段类型 [字段约束],
    字段名称2 字段类型 [字段约束],
    ...
) engine=innodb default charset=utf8;
```

### 1、主键约束

1、PRIMARY KEY 约束唯一标识数据库表中的每条记录。
2、主键必须包含唯一的值。
3、主键列不能包含 NULL 值。
4、每个表都应该有一个主键，并且每个表只能有一个主键。

遵循原则：

1）主键应当是对用户没有意义的
2）永远也不要更新主键。
3）主键不应包含动态变化的数据，如时间戳、创建时间列、修改时间列等。
4） 主键应当由计算机自动生成。

创建主键约束：创建表时，在字段描述处，声明指定字段为主键

![image-20210906183745166](media/image-20210906183745166.png)

案例：创建一个学生信息表tb_students，包含编号id、学生姓名name、年龄age、性别gender以及家庭住址address等字段，然后将id设置为主键。

```sql
create table tb_students(
	id int primary key,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男', '女'),
    address varchar(255)
) engine=innodb default charset=utf8;
```

删除主键约束：如需撤销 PRIMARY KEY 约束，请使用下面的 SQL

```powershell
alter table persons2 drop primary key;
```

![image-20210906183908865](media/image-20210906183908865.png)

案例：删除tb_students数据表的主键

```sql
alter table tb_students drop primary key;
```



> 补充：自动增长

我们通常希望在每次插入新记录时，数据库自动生成字段的值。

我们可以在表中使用 auto_increment（自动增长列）关键字，自动增长列类型必须是整型，自动增长列必须为键(一般是主键)。

**下列 SQL 语句把 "Persons" 表中的 "Id" 列定义为** **auto_increment** **主键**

```powershell
create table persons3(
	id int auto_increment primary key,
	first_name varchar(255),
	last_name varchar(255),
	address varchar(255),
	city varchar(255)
) default charset=utf8;
```

向persons添加数据时，可以不为Id字段设置值，也可以设置成null，数据库将自动维护主键值：

```powershell
insert into persons3(first_name,last_name) values('Bill','Gates');
insert into persons3(id,first_name,last_name) values(null,'Bill','Gates');
```

运行效果：

![image-20210906184220825](media/image-20210906184220825.png)

案例：创建一个学生信息表tb_students，包含编号id、学生姓名name、年龄age、性别gender以及家庭住址address等字段，然后将id设置为主键自动增长列。

```sql
drop table tb_students;

create table tb_students(
	id int auto_increment primary key,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男', '女'),
    address varchar(255)
) engine=innodb default charset=utf8;
或
create table tb_students(
	id int auto_increment,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男', '女'),
    address varchar(255),
    primary key(id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into tb_students values (null, '吕布', 30, '男', '内蒙古包头市');
insert into tb_students values (null, '貂蝉', 19, '女', '山西忻州市');
```

### 2、非空约束

NOT NULL 约束强制列不接受 NULL 值。
NOT NULL 约束强制字段始终包含值。这意味着，如果不向字段添加值，就无法插入新记录或者更新记录。
下面的 SQL 语句强制 "id" 列和 "last_name" 列不接受 NULL 值：

![image-20210906184938237](media/image-20210906184938237.png)

案例：创建一个tb_news新闻表，包含id主键列、title新闻标题、description描述、content新闻内容以及addtime添加时间，要求为title字段添加非空约束。

```sql
create table tb_news(
	id int auto_increment,
    title varchar(80) not null,
    description varchar(255),
    content text,
    addtime datetime,
    primary key(id)
) engine=innodb default charset=utf8;
```

### 3、唯一约束

UNIQUE 约束唯一标识数据库表中的每条记录。
UNIQUE 和 PRIMARY KEY 约束均为列或列集合提供了唯一性的保证。
PRIMARY KEY 拥有自动定义的 UNIQUE 约束。

请注意：
每个表可以有多个 UNIQUE 约束，但是每个表只能有一个 PRIMARY KEY 约束。

![image-20210906185040009](media/image-20210906185040009.png)

案例：创建一个tb_member会员表 ，包含字段有id主键、username用户名、password密码（密码必须使用密文保存，长度为固定的32位），由于用户名不允许出现重复的情况，所以请为username添加唯一约束。

```sql
create table tb_member(
	id int auto_increment,
    username varchar(20) unique,
    password char(32),
    primary key(id)
) engine=innodb default charset=utf8;
```

### 4、默认值约束

关键字：default

用来指定某列的默认值。在表中插入一条新记录时，如果没有为这个字段赋值，系统就会自动为这个字段插入默认值。

案例：创建一个tb_department部门表，包含字段id主键、name部门名称以及location部门位置。由于我们的部门位置位于北京的较多，所以部门位置就可以默认为“Beijing”。

```sql
create table tb_department(
	id int auto_increment,
    name varchar(20),
    location varchar(50) default 'Beijing',
    primary key(id)
) engine=innodb default charset=utf8;
```

### 5、外键约束(了解)

外键约束：关键字foreign key（主要用于==多表==关联使用）

比如：有两张数据表，这两个数据表之间==有联系==，通过了==某个字段==可以建立连接，这个字段在其中一个表中是==主键==，在另外一张表中，我们就把其称之为==外键==。

![image-20220222174336589](media/image-20220222174336589.png)

### 6、小结

① 主键约束：唯一标示，不能重复，不能为空。
1）主键应当是对用户没有意义的
2）永远也不要更新主键。
3）主键不应包含动态变化的数据，如时间戳、创建时间列、修改时间列等。
4） 主键应当由计算机自动生成。

自动增长：
我们可以在表中使用 auto_increment（自动增长列）关键字，自动增长列类型必须是整型，自动增长列必须为键(一般是主键)。

② 非空约束：
NOT NULL 约束强制列不接受 NULL 值。

③ 唯一约束：
UNIQUE 约束唯一标识数据库表中的每条记录。
UNIQUE 和 PRIMARY KEY 约束均为列或列集合提供了唯一性的保证。
PRIMARY KEY 拥有自动定义的 UNIQUE 约束。

④ 默认值约束

default 默认值

用来指定某列的默认值。在表中插入一条新记录时，如果没有为这个字段赋值，系统就会自动为这个字段插入默认值。

⑤ 外键约束（了解）

主要用于指定两张表之间的关联关系。

## 十、DQL数据查询语言

### 1、数据集准备

```mysql
CREATE TABLE product
(
    pid         INT PRIMARY KEY,
    pname       VARCHAR(20),
    price       DOUBLE,
    category_id VARCHAR(32)
);
```

插入数据：

```mysql
INSERT INTO product VALUES (1,'联想',5000,'c001’);
INSERT INTO product VALUES (2,'海尔',3000,'c001’);
INSERT INTO product VALUES (3,'雷神',5000,'c001’);
INSERT INTO product VALUES (4,'杰克琼斯',800,'c002’);
INSERT INTO product VALUES (5,'真维斯',200,'c002’);
INSERT INTO product VALUES (6,'花花公子',440,'c002’);
INSERT INTO product VALUES (7,'劲霸',2000,'c002’);
INSERT INTO product VALUES (8,'香奈儿',800,'c003’);
INSERT INTO product VALUES (9,'相宜本草',200,'c003’);
INSERT INTO product VALUES (10,'面霸',5,'c003’);
INSERT INTO product VALUES (11,'好想你枣',56,'c004’);
INSERT INTO product VALUES (12,'香飘飘奶茶',1,'c005’);
INSERT INTO product VALUES (13,'海澜之家',1,'c002');
```


### 2、select查询

基础查询：

```powershell
# 根据某些条件从某个表中查询指定字段的内容
格式：select [distinct]*| 列名,列名 from 表 where 条件
```

高级查询：SQL查询五子句（重点）

```sql
select */列名,列名 from 数据表 where 子句 group by 子句 having 子句 order by 子句 limit 子句;

① where子句
② group by子句
③ having子句
④ order by子句
⑤ limit子句
```

### 3、简单查询

```mysql
# 1.查询所有的商品.  
select *  from product;
# 2.查询商品名和商品价格. 
select pname,price from product;
# 3.查询结果是表达式（运算查询）：将所有商品的价格+10元进行显示.
select pname,price+10 from product;
```

### 4、条件查询

![image-20210906185658519](media/image-20210906185658519.png)

#### ☆ 比较查询

```powershell
# 查询商品名称为“花花公子”的商品所有信息：
SELECT * FROM product WHERE pname = '花花公子';
# 查询价格为800商品
SELECT * FROM product WHERE price = 800;
# 查询价格不是800的所有商品
SELECT * FROM product WHERE price != 800;
SELECT * FROM product WHERE price <> 800;
# 查询商品价格大于60元的所有商品信息
SELECT * FROM product WHERE price > 60;
# 查询商品价格小于等于800元的所有商品信息
SELECT * FROM product WHERE price <= 800;
```

#### ☆ 范围查询

```powershell
# 查询商品价格在200到1000之间所有商品
SELECT * FROM product WHERE price BETWEEN 200 AND 1000;
# 查询商品价格是200或800的所有商品
SELECT * FROM product WHERE price IN (200,800);
```

#### ☆ 逻辑查询

```powershell
# 查询商品价格在200到1000之间所有商品
SELECT * FROM product WHERE price >= 200 AND price <=1000;
# 查询商品价格是200或800的所有商品
SELECT * FROM product WHERE price = 200 OR price = 800;
# 查询价格不是800的所有商品
SELECT * FROM product WHERE NOT(price = 800);
```

#### ☆ 模糊查询

```powershell
# 查询以'香'开头的所有商品
SELECT * FROM product WHERE pname LIKE '香%';
# 查询第二个字为'想'的所有商品
SELECT * FROM product WHERE pname LIKE '_想%';
```

#### ☆ 非空查询

```powershell
# 查询没有分类的商品
SELECT * FROM product WHERE category_id IS NULL;
# 查询有分类的商品
SELECT * FROM product WHERE category_id IS NOT NULL;
```

### 5、排序查询

```powershell
# 通过order by语句，可以将查询出的结果进行排序。暂时放置在select语句的最后。
格式：SELECT * FROM 表名 ORDER BY 排序字段 ASC|DESC;
ASC 升序 (默认)
DESC 降序

# 1.使用价格排序(降序)
SELECT * FROM product ORDER BY price DESC;
# 2.在价格排序(降序)的基础上，以分类排序(降序)
SELECT * FROM product ORDER BY price DESC,category_id DESC;
```

### 6、聚合查询

之前我们做的查询都是横向查询，它们都是根据条件一行一行的进行判断，而使用聚合函数查询是纵向查询，它是对一列的值进行计算，然后返回一个单一的值；另外聚合函数会忽略空值。

今天我们学习如下五个聚合函数：

| **聚合函数** | **作用**                                                     |
| ------------ | ------------------------------------------------------------ |
| count()      | 统计指定列不为NULL的记录行数；                               |
| sum()        | 计算指定列的数值和，如果指定列类型不是数值类型，则计算结果为0 |
| max()        | 计算指定列的最大值，如果指定列是字符串类型，使用字符串排序运算； |
| min()        | 计算指定列的最小值，如果指定列是字符串类型，使用字符串排序运算； |
| avg()        | 计算指定列的平均值，如果指定列类型不是数值类型，则计算结果为0 |

案例演示：

```powershell
# 1、查询商品的总条数
SELECT COUNT(*) FROM product;
# 2、查询价格大于200商品的总条数
SELECT COUNT(*) FROM product WHERE price > 200;
# 3、查询分类为'c001'的所有商品的总和
SELECT SUM(price) FROM product WHERE category_id = 'c001';
# 4、查询分类为'c002'所有商品的平均价格
SELECT AVG(price) FROM product WHERE categ ory_id = 'c002';
# 5、查询商品的最大价格和最小价格
SELECT MAX(price),MIN(price) FROM product;
```

### 7、分组查询与having子句

#### ☆ 分组查询介绍

分组查询就是将查询结果按照指定字段进行分组，字段中数据相等的分为一组。

**分组查询基本的语法格式如下：**

GROUP BY 列名 [HAVING 条件表达式] [WITH ROLLUP]

**说明:**

- 列名: 是指按照指定字段的值进行分组。
- HAVING 条件表达式: 用来过滤分组后的数据。
- WITH ROLLUP：在所有记录的最后加上一条记录，显示select查询时聚合函数的统计和计算结果

#### ☆ group by的使用
创建数据集

```sql
create table students(
	id int auto_increment,
	name varchar(20),
	age tinyint unsigned,
	gender enum('male', 'female'),
	height float(5,2),
	primary key(id)
) engine=innodb default charset=utf8;

insert into students values (null,'郭靖',33,'male',1.80);
insert into students values (null,'黄蓉',19,'female',1.65);
insert into students values (null,'柯镇恶',45,'male',1.61);
insert into students values (null,'黄药师',50,'male',1.72);
insert into students values (null,'华筝',18,'female',1.60);
```



group by可用于单个字段分组，也可用于多个字段分组

```sql
-- 根据gender字段来分组
select gender from students group by gender;
-- 根据name和gender字段进行分组
select name, gender from students group by name, gender;
```

① group by可以实现去重操作

② group by的作用是为了实现分组统计（group by + 聚合函数）

#### ☆ group by + 聚合函数的使用

```sql
-- 统计不同性别的人的平均年龄
select gender,avg(age) from students group by gender;
-- 统计不同性别的人的个数
select gender,count(*) from students group by gender;
```

执行原理图

![image-20220225232628411](media/image-20220225232628411.png)

#### ☆ group by + having的使用

having作用和where类似都是过滤数据的，但having是过滤分组数据的，只能用于group by

```sql
-- 根据gender字段进行分组，统计分组条数大于2的
select gender,count(*) from students group by gender having count(*)>2;
```

案例演示：

```powershell
#1 统计各个分类商品的个数
SELECT category_id ,COUNT(*) FROM product GROUP BY category_id ;

#2 统计各个分类商品的个数,且只显示个数大于1的信息
SELECT category_id ,COUNT(*) FROM product GROUP BY category_id HAVING COUNT(*) > 1;
```

#### ☆ 扩展：with rollup回溯统计

之前的统计操作都是针对分组中的信息进行统计，那如果我们想在分组统计以后，在针对所有分组进行一个汇总，应该如何实现呢？

答：使用with rollup回溯统计

举个例子：针对students学生表，统计每个性别下的同学数量，如果这个时候，我们还想显示班级的总人数（分组数据汇总），就可以在这个SQL语句的后面一个with rollup了。

```sql
select gender,count(*) from students group by gender with rollup;
```
### 8、limit分页查询

作用：限制数据的查询数量

基本语法：

```sql
select * from 数据表 limit 查询数量;
```

案例：查询学生表中，身高最高的3名同学信息

```sql
select * from students order by height desc limit 3;
```



limit除了可以限制查询数量以外，其还可以指定从哪条数据开始查起，limit完整语法：

```sql
select * from students limit offset,count;

offset：索引，默认从0开始
count：查询总数量
```

案例：查询学生表中，身高第2、3高的同学信息

```sql
select * from students order by height desc limit 1,2;
```



limit子句典型应用场景：

分页查询在项目开发中常见，由于数据量很大，显示屏长度有限，因此对数据需要采取分页显示方式。例如数据共有30条，每页显示5条，第一页显示1-5条，第二页显示6-10条。

 格式：

```powershell
SELECT 字段1，字段2... FROM 表名 LIMIT M,N
M: 整数，表示从第几条索引开始，计算方式 （当前页-1）* 每页显示条数
N: 整数，表示查询多少条数据
SELECT 字段1，字段2... FROM 表名 LIMIT 0,5
SELECT 字段1，字段2... FROM 表名 LIMIT 5,5
```

### 9、小结

```powershell
SQL查询五子句：
select * from 表名 where子句 group by子句 having子句 order by子句 limit子句;
特别注意：查询五子句中，五子句的顺序一定要严格按照以上格式。

条件查询：SELECT *|字段名 FROM 表名 WHERE 条件；
排序查询：SELECT * FROM 表名 ORDER BY 排序字段 ASC|DESC;
聚合查询函数：count()，sum()，max()，min()，avg()。
分组查询：SELECT 字段1,字段2… FROM 表名 GROUP BY 分组字段 HAVING 分组条件;
分页查询：
SELECT 字段1，字段2... FROM 表名 LIMIT M,N
M: 整数，表示从第几条索引开始，计算方式 （当前页-1）*每页显示条数
N: 整数，表示查询多少条数据
```

## 十一、多表查询

### 数据集准备

classes班级表

```sql
create table classes(
	cls_id tinyint auto_increment,
    cls_name varchar(20),
    primary key(cls_id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into classes values (null, 'ui');
insert into classes values (null, 'java');
insert into classes values (null, 'python');
```

students学生表

```sql
create table students(
	id int auto_increment,
    name varchar(20),
    age tinyint unsigned,
    gender enum('male','female'),
    score float(5,1),
	cls_id tinyint,
    primary key(id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into students values (null,'刘备',34,'male',90.0,2);
insert into students values (null,'貂蝉',18,'female',75.0,1);
insert into students values (null,'赵云',28,'male',95.0,3);
insert into students values (null,'关羽',32,'male',98.0,3);
insert into students values (null,'大乔',19,'female',80.0,1);
```



### 交叉连接(了解)

没有意思，但是它是所有连接的基础。其功能就是将表1和表2中的每一条数据进行连接。

结果：

字段数 = 表1字段 + 表2的字段

记录数 = 表1中的总数量 * 表2中的总数量（笛卡尔积）

```powershell
select * from students cross join classes;
或
select * from students, classes;
```

![image-20210330160813460](media/image-20210330160813460.png)

### 1、内连接

#### ☆ 连接查询的介绍

连接查询可以实现多个表的查询，当查询的字段数据来自不同的表就可以使用连接查询来完成。

连接查询可以分为:

1. 内连接查询
2. 左外连接查询
3. 右外连接查询

#### ☆ 内连接查询

查询两个表中符合条件的共有记录

![image-20210329231722765](media/image-20210329231722765.png)

**内连接查询语法格式:**

```sql
select 字段 from 表1 inner join 表2 on 表1.字段1 = 表2.字段2
```

**说明:**

- inner join 就是内连接查询关键字
- on 就是连接查询条件

**例1：使用内连接查询学生表与班级表:**

```sql
select * from students as s inner join classes as c on s.cls_id = c.id;
```

#### ☆ 小结

- 内连接使用inner join .. on .., on 表示两个表的连接查询条件
- 内连接根据连接查询条件取出两个表的 “交集”

### 2、左外连接

#### ☆ 左连接查询

以左表为主根据条件查询右表数据，如果根据条件查询右表数据不存在使用null值填充

![image-20210329232043956](media/image-20210329232043956.png)

**左连接查询语法格式:**

```sql
select 字段 from 表1 left join 表2 on 表1.字段1 = 表2.字段2
```

**说明:**

- left join 就是左连接查询关键字
- on 就是连接查询条件
- 表1 是左表
- 表2 是右表

**例1：使用左连接查询学生表与班级表:**

```sql
select * from students as s left join classes as c on s.cls_id = c.id;
```

**例2：查询学生表中每一位学生（包括没有对应班级的学生）所属的班级信息**

前提：

在students学生表中，插入一条测试数据

```sql
insert into students values (null,'林黛玉',19,'female',96.0,99);
```

执行左外连接查询：

```sql
select * from students as s left join classes as c on s.cls_id = c.id;
```
#### ☆ 小结

- 左连接使用left join .. on .., on 表示两个表的连接查询条件
- 左连接以左表为主根据条件查询右表数据，右表数据不存在使用null值填充。

### 3、右外连接

#### ☆ 右连接查询

以右表为主根据条件查询左表数据，如果根据条件查询左表数据不存在使用null值填充

![image-20210329232137674](media/image-20210329232137674.png)

**右连接查询语法格式:**

```sql
select 字段 from 表1 right join 表2 on 表1.字段1 = 表2.字段2
```

**说明:**

- right join 就是右连接查询关键字
- on 就是连接查询条件
- 表1 是左表
- 表2 是右表

**例1：使用右连接查询学生表与班级表:**

```sql
select * from students as s right join classes as c on s.cls_id = c.id;
```

#### ☆ 小结

- 右连接使用right join .. on .., on 表示两个表的连接查询条件
- 右连接以右表为主根据条件查询左表数据，左表数据不存在使用null值填充。

## 十二、子查询(三步走)

### 1、子查询（嵌套查询）的介绍

在一个 select 语句中,嵌入了另外一个 select 语句, 那么被嵌入的 select 语句称之为子查询语句，外部那个select语句则称为主查询.

**主查询和子查询的关系:**

1. 子查询是嵌入到主查询中
2. 子查询是辅助主查询的,要么充当条件,要么充当数据源(数据表)
3. 子查询是可以独立存在的语句,是一条完整的 select 语句

### 2、子查询的使用

**例1. 查询学生表中大于平均年龄的所有学生:**

需求：查询年龄 > 平均年龄的所有学生

前提：① 获取所有学生的平均年龄

​			  ② 查询表中的所有记录，判断哪个同学 > 平均年龄值

第一步：写子查询

```powershell
select avg(age) from students;
```

第二步：写主查询

```powershell
select * from students where age > (平均值);
```

第三步：第一步和第二步进行合并

```powershell
select * from students where age > (select avg(age) from students);
```



**例2. 查询学生在班的所有班级名字:**

需求：显示所有有学生的班级名称

前提：① 先获取所有学员都属于那些班级

​	         ② 查询班级表中的所有记录，判断是否出现在①结果中，如果在，则显示，不在，则忽略。

第一步：编写子查询

```powershell
select distinct cls_id from students is not null;
```

第二步：编写主查询

```powershell
select * from classes where cls_id in (1, 2, 3);
```

第三步：把主查询和子查询合并

```sql
select * from classes where cls_id in (select distinct cls_id from students where cls_id is not null);
```



**例3. 查找年龄最小,成绩最低的学生:**

第一步：获取年龄最小值和成绩最小值

```powershell
select min(age), min(score) from student;
```

第二步：查询所有学员信息（主查询）

```sql
select * from students where (age, score) = (最小年龄, 最少成绩);
```

第三步：把第一步和第二步合并

```powershell
select * from students where (age, score) = (select min(age), min(score) from students);
```

### 3、小结

子查询是一个完整的SQL语句，子查询被嵌入到一对小括号里面

## 掌握子查询编写三步走