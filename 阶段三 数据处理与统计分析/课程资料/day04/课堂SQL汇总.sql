-- 1、选择数据库
use db_itheima;

-- 2、创建一个tb_students数据表
create table tb_students(
    id int primary key,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男','女'),
    address varchar(255)
) engine=innodb default charset=utf8;

desc tb_students;

-- 3、删除数据表中的主键列
alter table tb_students drop primary key;

-- 4、删除tb_students数据表
drop table tb_students;

-- 5、创建数据表并制定主键自动增长
create table tb_students(
    id int auto_increment primary key,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男','女'),
    address varchar(255)
) engine=innodb default charset=utf8;

-- 或
create table tb_students(
    id int auto_increment,
    name varchar(20),
    age tinyint unsigned,
    gender enum('男','女'),
    address varchar(255),
    primary key(id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into tb_students values (null, '吕布', 30, '男', '内蒙古包头市');
insert into tb_students values (null, '貂蝉', 19, '女', '山西忻州市');

select * from db_itheima.tb_students;

-- 6、创建一个tb_news数据表，将title字段添加为非空约束
create table tb_news(
    id int auto_increment,
    title varchar(80) not null,
    description varchar(255),
    content text,
    addtime datetime,
    primary key(id)
) engine=innodb default charset=utf8;

desc tb_news;

-- 7、创建一个tb_member会员表，将username字段设置为唯一键（唯一约束）
create table tb_member(
    id int auto_increment,
    username varchar(20) unique,
    password char(32),
    primary key(id)
) engine=innodb default charset=utf8;

desc tb_member;

-- 8、创建一个tb_department部门表，将location部门位置设置一个默认值为“Beijing”
create table tb_department(
    id int auto_increment,
    name varchar(20),
    location varchar(50) default "Beijing",
    primary key(id)
) engine=innodb default charset=utf8;

desc tb_department;

-- 9、准备product数据集
CREATE TABLE product
(
    pid         INT PRIMARY KEY,
    pname       VARCHAR(20),
    price       DOUBLE,
    category_id VARCHAR(32)
) DEFAULT CHARSET=utf8;

-- 插入测试数据
INSERT INTO product VALUES (1,'联想',5000,'c001');
INSERT INTO product VALUES (2,'海尔',3000,'c001');
INSERT INTO product VALUES (3,'雷神',5000,'c001');
INSERT INTO product VALUES (4,'杰克琼斯',800,'c002');
INSERT INTO product VALUES (5,'真维斯',200,'c002');
INSERT INTO product VALUES (6,'花花公子',440,'c002');
INSERT INTO product VALUES (7,'劲霸',2000,'c002');
INSERT INTO product VALUES (8,'香奈儿',800,'c003');
INSERT INTO product VALUES (9,'相宜本草',200,'c003');
INSERT INTO product VALUES (10,'面霸',5,'c003');
INSERT INTO product VALUES (11,'好想你枣',56,'c004');
INSERT INTO product VALUES (12,'香飘飘奶茶',1,'c005');
INSERT INTO product VALUES (13,'海澜之家',1,'c002');

-- 10、SQL基础查询
select * from product;
select pname,price from product;
select pname,price+10 from product;

-- 11、SQL五子句中的where子句之条件查询（结合比较运算符）
# 查询商品名称为“花花公子”的商品所有信息
select * from product where pname='花花公子';
# 查询价格为800商品
select * from product where price=800;
# 查询价格不是800的所有商品
select * from product where price<>800;
select * from product where price!=800;
# 查询商品价格大于60元的所有商品信息
select * from product where price>60;
# 查询商品价格小于等于800元的所有商品信息
select * from product where price<=800;

-- 12、SQL五子句中的where子句之条件查询（结合范围查询）
# 查询商品价格在200到1000之间所有商品
select * from product where price between 200 and 1000;
# 查询商品价格是200或800的所有商品
select * from product where price in (200,800);

-- 13、SQL五子句中的where子句之条件查询（结合like实现模糊查询）
# 查询以'香'开头的所有商品
select * from product where pname like '香%';
# 查询第二个字为'想'的所有商品
select * from product where pname like '_想%';

-- 14、SQL五子句中的where子句之条件查询（结合is null或is not null实现空值与非空值判断）
# 查询没有分类的商品
select * from product where category_id is null;
# 查询有分类的商品
select * from product where category_id is not null;

-- 15、SQL五子句中的where子句之条件查询（结合逻辑运算符实现逻辑查询）
# 查询商品价格在200到1000之间所有商品
select * from product where price >= 200 and price <= 1000;
# 查询商品价格是200或800的所有商品
select * from product where price = 200 or price = 800;
# 查询价格不是800的所有商品
select * from product where not(price = 800);

-- 16、order by排序查询（记住两个关键字asc升序排列与desc降序排列）
# 1.使用价格排序(降序)
select * from product order by price desc;
# 2.在价格排序(降序)的基础上，以分类排序(降序)，执行原理：首先会按照第一个字段进行排序，如果值相同，则按照第二个字段进行排序，依次类推
select * from product order by price desc, category_id desc;

-- 17、聚合查询
# 1、查询商品的总条数
select count(*) from product;
# 2、查询价格大于200商品的总条数
select count(*) from product where price > 200;
# 3、查询分类为'c001'的所有商品的总和
select sum(price) from product where category_id = 'c001';
# 4、查询分类为'c002'所有商品的平均价格
select avg(price) from product where category_id = 'c002';
# 5、查询商品的最大价格和最小价格
select max(price),min(price) from product;

-- 18、创建分组数据集
create table students(
	id int auto_increment,
	name varchar(20),
	age tinyint unsigned,
	gender enum('male', 'female'),
	height float(5,2),
	primary key(id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into students values (null,'郭靖',33,'male',1.80);
insert into students values (null,'黄蓉',19,'female',1.65);
insert into students values (null,'柯镇恶',45,'male',1.61);
insert into students values (null,'黄药师',50,'male',1.72);
insert into students values (null,'华筝',18,'female',1.60);

select * from students;

-- 根据gender字段来分组
select gender from students group by gender;

-- 根据name和gender字段进行分组
select name, gender from students group by name, gender;

-- 19、分组与聚合函数结合使用案例
-- 统计不同性别的人的平均年龄
select gender,avg(age) from students group by gender;
-- 统计不同性别的人的个数
select gender,count(*) from students group by gender;

-- 20、having子句的作用：简单查询中，having与where子句功能基本类似，都可以实现数据过滤
-- 查询students学生表年龄大于30的同学信息
select * from students where age > 30;
select * from students having age > 30;

-- 21、group by + having的使用
-- 根据gender字段进行分组，统计分组条数大于2的分组信息
select gender,count(*) from students group by gender having count(*) > 2;

#1 统计各个分类商品的个数
select category_id,count(*) from product group by category_id;

#2 统计各个分类商品的个数,且只显示个数大于1的信息
select category_id,count(*) from product group by category_id having count(*) > 1;

-- 22、with rollup回溯统计
select gender,count(*) from students group by gender with rollup;

-- 23、limit子句使用
-- 案例：查询学生表中，身高最高的3名同学信息
select * from students order by height desc limit 3;

-- 案例：查询学生表中，身高第2、3高的同学信息
select * from students order by height desc limit 1,2;

-- 24、准备多表查询的数据集
drop table classes;
drop table students;

create table classes(
	cls_id tinyint auto_increment,
    cls_name varchar(20),
    primary key(cls_id)
) engine=innodb default charset=utf8;

-- 插入测试数据
insert into classes values (null, 'ui');
insert into classes values (null, 'java');
insert into classes values (null, 'python');

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

-- 25、交叉连接查询
select * from students cross join classes;
select * from students, classes;

-- 26、内连接查询，查询学生表中每个学生所属的班级信息
select * from students inner join classes on students.cls_id = classes.cls_id;
select students.*, classes.cls_name from students inner join classes on students.cls_id = classes.cls_id;
-- 27、数据表别名与字段别名，在SQL中，数据表 [as] 别名，字段 [as] 别名
select s.*,c.cls_name as class_name from students s inner join classes c on s.cls_id = c.cls_id;

-- 28、外连接查询中的左外连接查询，查询学生表中每个学生所属的班级信息
select * from students s left join classes c on s.cls_id = c.cls_id;
-- 插入一条测试数据
insert into students values (null,'林黛玉',19,'female',96.0,99);
select * from students s left join classes c on s.cls_id = c.cls_id;

-- 29、外连接中的右外连接查询，查询学生表中每个学生所属的班级信息
select * from classes c right join students s on c.cls_id = s.cls_id;

-- 30、子查询操作
-- 案例1：查询学生表中大于平均年龄的所有学生
select * from students where age > (select avg(age) from students);
-- 案例2：查询有学生的班级信息
select * from classes where cls_id in (select distinct cls_id from students where cls_id is not null);
-- 案例3：查找年龄最小,成绩最低的学生
select * from students where (age, score) = (select min(age), min(score) from students);