# 2 MySQL基础II

## 6.DML数据操作语言

### 6.1.插入表记录：insert

- 语法：

  ```mysql
  -- 向表中插入某些字段
  insert into 表 (字段1,字段2,字段3..) values(值1,值2,值3..);
  -- 向表中插入所有字段,字段的顺序为创建表时的顺序
  insert into 表 values(值1,值2,值3..)
  ```

注意：
  - 值与字段必须对应，个数相同，类型相同

  - 值的数据大小必须在字段的长度范围内

  - 除了数值类型外，其它的字段类型的值必须使用引号引起。（建议单引号）'' &quot;&quot;``

  - 如果要插入空值，可以不写字段，或者插入 null。

- 例如：

  ```mysql
  INSERT INTO category(cid,cname) VALUES('c001','电器'); 
  INSERT INTO category(cid,cname) VALUES('c002','服饰'); 
  INSERT INTO category(cid,cname) VALUES('c003','化妆品'); 
  INSERT INTO category(cid,cname) VALUES('c004','书籍');
  INSERT INTO category(cid) VALUES('c005'); 
  insert into category values('06','玩具'),('07','蔬菜');
  ```

### 6.2.更新表记录：update

用来修改指定条件的数据，将满足条件的记录指定列修改为指定值

- 语法：

  ```mysql
  -- 更新所有记录的指定字段
  update 表名 set 字段名=值,字段名=值,...;
  -- 更新符号条件记录的指定字段
  update 表名 set 字段名=值,字段名=值,... where 条件;
  ```

- 示例:

  ```mysql
  update category set cname  = '家电';  #将所有行的cname改为'家电'
  update category set cname  = '水果' where cid = 'c001'; #将cid为c001的cname修改为水果
  ```

注意：

- 列名的类型与修改的值要一致.
- 修改值得时候不能超过最大长度.
- 除了 **数值类** 型外，其它的字段类型的值必须使用引号引起

### 6.3.删除记录：delete 或 truncate

- 语法：

  ```mysql
  delete from 表名 [where 条件];
  或者
  truncate table 表名;
  ```

- 示例:

  ```mysql
  delete from category where cid = '005'; #删除cid为005的纪录
  truncate table category;  #清空表数据
  ```

注意:

- delete 一条一条删除，不清空auto_increment记录数。

- truncate 直接将表删除，重新建表，auto_increment将置为零，从新开始。

- 除此之外：delete 和 truncate 还有以下区别：
  1. Delete 可以按照行来删除，truncate 只能全表删除；
  2. Delete 是DML 操作，truncate 是DDL 操作；
  3. Delete 在删除前会锁行，truncate 删除前会锁表；
  4. Truncate 删除比 delete 要快，不需要写日志；
  5. Truncate 删除之后就不能恢复了， delete 可以恢复。

## 7.SQL约束

### 7.1.主键约束

**PRIMARY KEY** 约束唯一标识数据库表中的每条记录。

主键必须包含唯一的值。

主键列不能包含 NULL 值。

每个表都应该有一个主键，并且每个表只能有一个主键。

#### 7.1.1.添加主键约束3种

- 方式一：创建表时，在字段描述处，声明指定字段为主键：

  ```mysql
  CREATE TABLE Persons1(
   Id int PRIMARY KEY,
   LastName varchar(255),
   FirstName varchar(255),
   Address varchar(255),
   City varchar(255)
  )
  ```

- 方式二：创建表时，在constraint约束区域，声明指定字段为主键：
  - 格式：[constraint 名称] primary key (字段列表)
  - 关键字constraint可以省略，如果需要为主键命名，constraint不能省略，主键名称一般没用。
  - 字段列表需要使用小括号括住，如果有多字段需要使用逗号分隔。声明两个以上字段为主键，我们称为联合主键。
  
  ```mysql
  CREATE TABLE persons2
  (
    FirstName varchar(255),
    LastName  varchar(255),
    Address   varchar(255),
    City      varchar(255),
    CONSTRAINT pk_PersonID PRIMARY KEY (FirstName, LastName)
  )
  ```
  
  或
  
  ```mysql
  CREATE TABLE persons3
  (
    FirstName varchar(255) ,
    LastName  varchar(255),
    Address   varchar(255),
    City      varchar(255)
  )
  Alter table persons3 add primary key(firstname,lastname);
  ```

#### 7.1.2.删除主键约束

如需撤销PRIMARY KEY 约束，请使用下面的SQL：

```mysql
ALTER TABLE persons DROP PRIMARY KEY;
```

#### 7.1.3.自动增长列

我们通常希望在每次插入新记录时，数据库自动生成字段的值。

我们可以在表中使用auto_increment（自动增长列）关键字，自动增长列类型必须是整形，自动增长列必须为键(一般是主键)。

下列 SQL 语句把 &quot;persons&quot; 表中的 &quot;Id&quot; 列定义为 auto_increment 主键

```mysql
CREATE TABLE persons4
(
  Id      int PRIMARY KEY AUTO_INCREMENT,
  LastName  varchar(255),
  FirstName varchar(255),
  Address   varchar(255),
  City      varchar(255)
)
```

向persons添加数据时，可以不为Id字段设置值，也可以设置成null，数据库将自动维护主键值：

```mysql
INSERT INTO persons (FirstName,LastName) VALUES ('Bill','Gates')
INSERT INTO  persons (Id,FirstName,LastName) VALUES (NULL,'Bill','Gates')
```

扩展：默认AUTO_INCREMENT 的开始值是 1，如果希望修改起始值，请使用下列 SQL 语法：

```mysql
ALTER TABLE persons AUTO_INCREMENT=100
```

### 7.2.非空约束

NOT NULL 约束强制列不接受NULL 值。

NOT NULL 约束强制字段始终包含值。这意味着，如果不向字段添加值，就无法插入新记录或者更新记录。

下面的 SQL 语句强制 &quot;Id&quot; 列和 &quot;LastName&quot; 列不接受 NULL 值：

```mysql
CREATE TABLE persons5
(
  Id          int          NOT NULL,
  LastName  varchar(255) NOT NULL,
  FirstName  varchar(255),
  Address    varchar(255),
  City        varchar(255)
)
```

### 7.3.唯一约束

UNIQUE 约束唯一标识数据库表中的每条记录。

UNIQUE 和PRIMARY KEY 约束均为列或列集合提供了唯一性的保证。

PRIMARY KEY 拥有自动定义的UNIQUE 约束。

请注意，每个表可以有多个UNIQUE 约束，但是每个表只能有一个PRIMARY KEY 约束。

**添加唯一约束**

创建表时，在字段描述处，声明唯一：

```mysql
CREATE TABLE persons(
  Id      int UNIQUE,
  LastName  varchar(255) NOT NULL,
  FirstName varchar(255),
  Address   varchar(255),
  City      varchar(255)
)
```

### 7.4.外键约束

FOREIGN KEY 表示外键约束，将在多表中学习。

### 7.5.获取有关数据库和表的信息

在数据库列表中会有 information_schema 数据库，可以查看下面有很多表，这些表有下面两种类型的元数据。

如何操作呢？

```mysql
MySQL> USE INFORMATION_SCHEMA;
MySQL> show tables;
MySQL> DESC INFORMATION_SCHEMA.TABLES;
#如果想知道某个表的所有列及其定义
MySQL> SELECT * FROM COLUMNS WHERE TABLE_NAME='category'\G;
```