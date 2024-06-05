# Python高级语法与正则表达式

# 学习目标

1、能够掌握with语句的使用

2、能够知道生成器的两种创建方式

3、能够知道深拷贝和浅拷贝的区别

4、能够掌握Python中的正则表达式编写

# 一、Python高级语法

## 1、with语句和上下文管理器

### ☆ with语句

Python提供了 with 语句的写法，既简单又安全。

文件操作的时候使用with语句可以自动调用关闭文件操作，即使出现异常也会自动关闭文件操作。

举个栗子：

![image-20210119221959274](media/image-20210119221959274-1065999.png)

使用with方法实现文件操作，如下所示：

```python
# 1、以写的方式打开文件
with open('1.txt', 'w') as f:
    # 2、读取文件内容
    f.write('hello world')
```

## 2、生成器的创建方式

根据程序设计者制定的规则循环生成数据，当条件不成立时则生成数据结束

数据不是一次性全部生成出来，而是使用一个，再生成一个，可以节约大量的内存。

![image-20210120144636687](media/image-20210120144636687-1125196.png)

创建生成器的方式

① 生成器推导式

② yield 关键字

### ☆ 生成器推导式

与列表推导式类似，只不过生成器推导式使用小括号。

```python
# 创建生成器
my_generator = (i * 2 for i in range(5))
print(my_generator)

# next获取生成器下一个值
# value = next(my_generator)
# print(value)

# 遍历生成器
for value in my_generator:
    print(value)
```

生成器相关函数：

```powershell
next 函数获取生成器中的下一个值
for  循环遍历生成器中的每一个值 
```

### ☆ yield生成器

yield 关键字生成器的特征：在def函数中具有yield关键字

```python
def generator(n):
    for i in range(n):
        print('开始生成...')
        yield i
        print('完成一次...')
        
g = generator(5)
print(next(g))
print(next(g))
print(next(g))
print(next(g))
print(next(g))				----->    正常
print(next(g))  			----->    报错
Traceback (most recent call last):
  File "/Users/cndws/PycharmProjects/pythonProject/demo.py", line 14, in <module>
    print(next(g))
StopIteration
```

```python
def generator(n):
    for i in range(n):
        print('开始生成...')
        yield i
        print('完成一次...')
        
g = generator(5)
for i in g:
    print(i)
```

```python
def generator(n):
    for i in range(n):
        print('开始生成...')
        yield i
        print('完成一次...')
        
g = generator(5)
while True:
    try:
        print(next(g))
    except StopIteration:
        break
```

注意点：

① 代码执行到 yield 会暂停，然后把结果返回出去，下次启动生成器会在暂停的位置继续往下执行

② 生成器如果把数据生成完成，再次获取生成器中的下一个数据会抛出一个StopIteration 异常，表示停止迭代异常

③ while 循环内部没有处理异常操作，需要手动添加处理异常操作

④ for 循环内部自动处理了停止迭代异常，使用起来更加方便，推荐大家使用。

### ☆ yield关键字和return关键字

如果不太好理解`yield`，可以先把`yield`当作`return`的同胞兄弟来看，他们都在函数中使用，并履行着返回某种结果的职责。

这两者的区别是：

有`return`的函数直接返回所有结果，程序终止不再运行，并销毁局部变量；

```python
def example():
    x = 1
    return x

example = example()
print(example)
```

而有`yield`的函数则返回一个可迭代的 generator（生成器）对象，你可以使用for循环或者调用next()方法遍历生成器对象来提取结果。

```python
def example():
    x = 1
    y = 10
    while x < y:
        yield x
        x += 1

example = example()
print(example)
```



![image-20210120164257050](media/image-20210120164257050-1132177.png)

### ☆ 为什么要使用yield生成器

```python
import memory_profiler as mem


# nums = [1, 2, 3, 4, 5]
# print([i*i for i in nums])


nums = list(range(10000000))
print('运算前内存：', mem.memory_usage())
# 列表
# square_nums = [n * n for n in nums]
# 生成器
square_nums = (n * n for n in nums)
print('运算后内存：', mem.memory_usage())
```

### ☆ yield与斐波那契数列

数学中有个著名的斐波拉契数列（Fibonacci）

要求：数列中第一个数为0，第二个数为1，其后的每一个数都可由前两个数相加得到：

例子：1, 1, 2, 3, 5, 8, 13, 21, 34, ...

现在我们使用生成器来实现这个斐波那契数列，每次取值都通过算法来生成下一个数据, ==生成器每次调用只生成一个数据，可以节省大量的内存。==

```python
def fib(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fib(5): 
    print n
```

## 3、深浅拷贝

### ☆ 几个概念

- 变量：是一个系统表的元素，拥有指向对象的连接空间
- 对象：被分配的一块内存，存储其所代表的值
- 引用：是自动形成的从变量到对象的指针
- 类型：属于对象，而非变量
- 不可变对象：一旦创建就不可修改的对象，包括数值类型、字符串、布尔类型、元组

*（该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把原来的值复制一份后再改变，这会开辟一个新的地址，变量再指向这个新的地址。）*

- 可变对象：可以修改的对象，包括列表、字典、集合

*（该对象所指向的内存中的值可以被改变。变量（准确的说是引用）改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的地址，通俗点说就是原地改变。）*

当我们写：

```python
a = "python"
```

Python解释器干的事情：

① 创建变量a

② 创建一个对象(分配一块内存)，来存储值 'python'

③ 将变量与对象，通过指针连接起来，从变量到对象的连接称之为引用(变量引用对象)

![image-20210121111247319](media/image-20210121111247319-1198767.png)

### ☆ 赋值

**赋值: 只是复制了新对象的引用，不会开辟新的内存空间。**

并不会产生一个独立的对象单独存在，只是将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生变化，另一个标签也会随之改变。

![image-20211128145849883](media/image-20211128145849883.png)

### ☆ 浅拷贝

**浅拷贝: 创建新对象，其内容是原对象的引用。**

浅拷贝之所以称为浅拷贝，是它仅仅只拷贝了一层，拷贝了最外围的对象本身，内部的元素都只是拷贝了一个引用而已。

案例1：赋值

![image-20210121124612944](media/image-20210121124612944-1204373.png)

案例2：可变类型浅拷贝

![image-20210121124816952](media/image-20210121124816952-1204497.png)

案例3：不可变类型浅拷贝

![image-20210121124927773](media/image-20210121124927773-1204567.png)

> 注：不可变类型进行浅拷贝不会给拷贝的对象开辟新的内存空间，而只是拷贝了这个对象的引用



浅拷贝有三种形式： 切片操作，工厂函数，copy模块中的copy函数。

如： lst = [1,2,[3,4]]

切片操作：lst1 = lst[:] 或者 lst1 = [each for each in lst]

> 注：`[:]`它与`[0:]`相似，意思是从0索引拆分到末尾。它返回一个新列表。

工厂函数：lst1 = list(lst)

copy函数：lst1 = copy.copy(lst)

但是在lst中有一个嵌套的list[3,4]，如果我们修改了它，情况就不一样了。



浅复制要分两种情况进行讨论：

1）当浅复制的值是不可变对象（字符串、元组、数值类型）时和“赋值”的情况一样，对象的id值*（id()函数用于获取对象的内存地址）*与浅复制原来的值相同。

2）当浅复制的值是可变对象（列表、字典、集合）时会产生一个“不是那么独立的对象”存在。有两种情况：

*第一种情况：复制的对象中无复杂子对象，原来值的改变并不会影响浅复制的值，同时浅复制的值改变也并不会影响原来的值。原来值的id值与浅复制原来的值不同。*

*第二种情况：复制的对象中有复杂子对象（例如列表中的一个子元素是一个列表），如果不改变其中复杂子对象，浅复制的值改变并不会影响原来的值。 但是改变原来的值中的复杂子对象的值会影响浅复制的值。*

### ☆ 深拷贝

**深拷贝：和浅拷贝对应，深拷贝拷贝了对象的所有元素，包括多层嵌套的元素。深拷贝出来的对象是一个全新的对象，不再与原来的对象有任何关联。**

所以改变原有被复制对象不会对已经复制出来的新对象产生影响。只有一种形式，copy模块中的deepcopy函数。

可变类型深拷贝：

![image-20210121125125964](media/image-20210121125125964-1204686.png)

不可变类型深拷贝：不可变类型进行深拷贝不会给拷贝的对象开辟新的内存空间，而只是拷贝了这个对象的引用

![image-20210121125301655](media/image-20210121125301655-1204781.png)

### ☆ 案例演示

案例1：对于可变对象深浅拷贝

```python
import copy
a=[1,2,3]

print("=====赋值=====")
b=a
print(a)
print(b)
print(id(a))
print(id(b))

print("=====浅拷贝=====")
b=copy.copy(a)
print(a)
print(b)
print(id(a))
print(id(b))

print("=====深拷贝=====")
b=copy.deepcopy(a)
print(a)
print(b)
print(id(a))
print(id(b))
```

结果：

```python
=====赋值=====
[1, 2, 3]
[1, 2, 3]
37235144
37235144
=====浅拷贝=====
[1, 2, 3]
[1, 2, 3]
37235144
37191432
=====深拷贝=====
[1, 2, 3]
[1, 2, 3]
37235144
37210184
```

小结：

赋值： 值相等，地址相等

copy浅拷贝：值相等，地址不相等

deepcopy深拷贝：值相等，地址不相等



案例2：对于可变对象深浅拷贝（外层改变元素）

```python
import copy
l=[1,2,3,[4, 5]]

l1=l #赋值
l2=copy.copy(l) #浅拷贝
l3=copy.deepcopy(l) #深拷贝
l.append(6)

print(l)  
print(l1)
print(l2)
print(l3)
```

结果：

```python
[1, 2, 3, [4, 5], 6]     #l添加一个元素6
[1, 2, 3, [4, 5], 6]     #l1跟着添加一个元素6
[1, 2, 3, [4, 5]]        #l2保持不变
[1, 2, 3, [4, 5]]        #l3保持不变
```



案例3：对于可变对象深浅拷贝（内层改变元素）

```python
import copy
l=[1,2,3,[4, 5]]

l1=l #赋值
l2=copy.copy(l) #浅拷贝
l3=copy.deepcopy(l) #深拷贝
l[3].append(6) 

print(l) 
print(l1)
print(l2)
print(l3)
```

结果：

```python
[1, 2, 3, [4, 5, 6]]      #l[3]添加一个元素6
[1, 2, 3, [4, 5, 6]]      #l1跟着添加一个元素6
[1, 2, 3, [4, 5, 6]]      #l2跟着添加一个元素6
[1, 2, 3, [4, 5]]         #l3保持不变
```

小结：

① 外层添加元素时，浅拷贝不会随原列表变化而变化；内层添加元素时，浅拷贝才会变化。
② 无论原列表如何变化，深拷贝都保持不变。
③ 赋值对象随着原列表一起变化。



# 二、正则表达式概述

## 1、为什么要学习正则表达式

在实际开发过程中经常会有查找符合某些复杂规则的字符串的需要

比如：邮箱、图片地址、手机号码等

这时候想<font color="red">匹配或者查找符合某些规则的字符串</font>就可以使用==正则表达式==了

![image-20210118135358176](media/image-20210118135358176-0949238.png)

## 2、什么是正则表达式

正则表达式(regular expression)描述了一种字符串匹配的==模式==，可以用来检查一个串是否含有==某种==子串、将匹配的子串做替换或者从某个串中取出符合某个条件的子串等。 

模式：一种特定的字符串模式，这个模式是通过一些特殊的符号组成的。
某种：也可以理解为是一种模糊匹配。

精准匹配：select * from blog where title='python';

模糊匹配：select * from blog where title like ‘%python%’;

正则表达式并不是Python所特有的，在Java、PHP、Go以及JavaScript等语言中都是支持正则表达式的。

## 3、正则表达式的功能

① 数据验证（表单验证、如手机、邮箱、IP地址）
② 数据检索（数据检索、数据抓取）
③ 数据隐藏（135****6235 王先生）
④ 数据过滤（论坛敏感关键词过滤）
…

# 二、re模块的介绍

## 1、什么是re模块

在Python中需要通过正则表达式对字符串进行匹配的时候，可以使用一个re模块

## 2、re模块使用三步走

```python
# 第一步：导入re模块
import re
# 第二步：使用match方法进行匹配操作
result = re.match(pattern正则表达式, string要匹配的字符串, flags=0)
# 第三步：如果数据匹配成功，使用group方法来提取数据
result.group()
```

match函数参数说明：

| 参数    | 描述                                                         |
| ------- | ------------------------------------------------------------ |
| pattern | 匹配的正则表达式                                             |
| string  | 要匹配的字符串。                                             |
| flags   | 标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志 |

匹配成功re.match方法返回一个匹配的对象，否则返回None。

我们可以使用group(num) 或 groups() 匹配对象函数来获取匹配数据。



正则表达式可以包含一些可选标志修饰符来控制匹配的模式。修饰符被指定为一个可选的标志。多个标志可以通过按位 OR(|) 它们来指定。如 re.I | re.M 被设置成 I 和 M 标志：

| 修饰符 | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| re.I   | ==使匹配对大小写不敏感==                                     |
| re.L   | 做本地化识别（locale-aware）匹配，这个功能是为了支持多语言版本的字符集使用环境的，比如在转义符\w，在英文环境下，它代表[a-zA-Z0-9_]，即所以英文字符和数字。如果在一个法语环境下使用，缺省设置下，不能匹配"é" 或   "ç"。加上这L选项和就可以匹配了。不过这个对于中文环境似乎没有什么用，它仍然不能匹配中文字符。 |
| re.M   | ==多行匹配，影响 ^ 和 $==                                    |
| re.S   | ==使 . 匹配包括换行在内的所有字符==                          |
| re.U   | 根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.      |
| re.X   | VERBOSE，冗余模式， 此模式忽略正则表达式中的空白和#号的注释，例如写一个匹配邮箱的正则表达式。该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。 |

## 3、re模块的相关方法

### ☆ re.match(pattern, string, flags=0)

* 从字符串的起始位置匹配，如果匹配成功则返回匹配内容， 否则返回None

### ☆ re.findall(pattern, string, flags=0)

- 扫描整个串，返回所有与pattern匹配的列表
- 注意: 如果pattern中有分组则返回与分组匹配的列表
- 举例： `re.findall("\d","chuan1zhi2") >> ["1","2"]`

### ☆ re.finditer(pattern, string, flags)

* 功能与上面findall一样，不过返回的时迭代器



参数说明：

- pattern : 模式字符串。
- repl : 替换的字符串，也可为一个函数。
- string : 要被查找替换的原始字符串。
- count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
- flags: 匹配方式:
  - re.I 使匹配对大小写不敏感，I代表Ignore忽略大小写
  - re.S 使 . 匹配包括换行在内的所有字符
  - re.M 多行模式,会影响^,$

## 4、正则表达式快速入门

案例1：查找一个字符串中是否具有数字“8”

```python
import re


result = re.findall('8', '13566128753')
# print(result)
if result:
    print(result)
else:
    print('未匹配到任何数据')
```

案例2：查找一个字符串中是否具有数字

```python
import re


result = re.findall('\d', 'a1b2c3d4f5')
# print(result)
if result:
    print(result)
else:
    print('未匹配到任何数据')
```

案例3：查找一个字符串中是否具有非数字

```python
import re


result = re.findall('\D', 'a1b2c3d4f5')
# print(result)
if result:
    print(result)
else:
    print('未匹配到任何数据')
```

# 三、正则表达式详解

## 正则编写三步走：查什么、查多少、从哪查

## 1、查什么

| 代码 | 功能                      |
| ---- | ------------------------- |
| .（英文点号） | 匹配任意1个字符（除了\n） |
| [ ]  | 匹配[ ]中列举的某个字符，专业名词 => 字符簇 |
| \[^指定字符] | 匹配除了指定字符以外的其他某个字符，^专业名词 => 托字节 |
| \d   | 匹配数字，即0-9           |
| \D   | 匹配非数字，即不是数字    |
| \s   | 匹配空白，即   空格，tab键               |
| \S   | 匹配非空白                               |
| \w   | 匹配非特殊字符，即a-z、A-Z、0-9、_ |
| \W   | 匹配特殊字符，即非字母、非数字、非下划线 |

字符簇常见写法：

① [abcdefg] 代表匹配abcdefg字符中的任意某个字符（1个）

② [aeiou] 代表匹配a、e、i、o、u五个字符中的任意某个字符

③ [a-z] 代表匹配a-z之间26个字符中的任意某个

④ [A-Z] 代表匹配A-Z之间26个字符中的任意某个

⑤ [0-9] 代表匹配0-9之间10个字符中的任意某个

⑥ [0-9a-zA-Z] 代表匹配0-9之间、a-z之间、A-Z之间的任意某个字符



字符簇 + 托字节结合代表取反的含义：

① \[^aeiou] 代表匹配除了a、e、i、o、u以外的任意某个字符

② \[^a-z] 代表匹配除了a-z以外的任意某个字符



\d 等价于 [0-9]， 代表匹配0-9之间的任意数字

\D 等价于 \[^0-9]，代表匹配非数字字符，只能匹配1个

## 2、查多少

| 代码  | 功能                                                         |
| ----- | ------------------------------------------------------------ |
| *     | 匹配前一个字符出现0次或者无限次，即可有可无（0到多）         |
| +     | 匹配前一个字符出现1次或者无限次，即至少有1次（1到多）        |
| ?     | 匹配前一个字符出现1次或者0次，即要么有1次，要么没有（0或1）  |
| {m}   | 匹配前一个字符出现m次，匹配手机号码\d{11}                    |
| {m,}  | 匹配前一个字符至少出现m次，\\w{3,}，代表前面这个字符最少要出现3次，最多可以是无限次 |
| {m,n} | 匹配前一个字符出现从m到n次，\w{6,10}，代表前面这个字符出现6到10次 |

基本语法：

正则匹配字符.或\w或\S + 跟查多少

如\w{6, 10}

如.*，匹配前面的字符出现0次或多次

## 3、从哪查

| 代码 | 功能                 |
| ---- | -------------------- |
| ^    | 匹配以某个字符串开头 |
| $    | 匹配以某个字符串结尾 |



扩展：正则工具箱

http://jsrun.net/app/reg

http://tools.jb51.net/regex/create_reg

http://www.dycan.cn/Tools/js_expression/?&rand=d899032e86733fc430b087ebd5da1815

# 四、几个重要概念

## 作用

<img src="./images/1.jpg">

re.match('src="(.*)"', str)

src="./images/1.jpg"

分组获取的结果（捕获） => ./images/1.jpg

## 1、子表达式（又称之为分组）

在正则表达式中，通过一对圆括号括起来的内容，我们就称之为"子表达式"。

```powershell
re.search(r'\d(\d)(\d)', 'abcdef123ghijklmn')

注意：Python正则表达式前的 r 表示原生字符串（rawstring），该字符串声明了引号中的内容表示该内容的原始含义，避免了多次转义造成的反斜杠困扰。
```

正则表达式中\d\d\d中，(\\d)(\d)就是子表达式，一共有两个`()`圆括号，则代表两个子表达式

> 说明：findall方法，如果pattern中有分组则返回与分组匹配的列表，所以分组操作中不适合使用findall方法，建议使用search(匹配一个)或finditer(匹配多个)方法。

## 2、捕获

当正则表达式在字符串中匹配到相应的内容后，计算机系统会自动把子表达式所匹配的到内容放入到系统的对应缓存区中（缓存区从$1开始）

![image-20210118194614636](media/image-20210118194614636-0970374.png)

案例演示：

```python
import re


# 匹配字符串中连续出现的两个相同的单词
str1 = 'abcdef123ghijklmn'
result = re.search(r'\d(\d)(\d)', str1)
print(result.group())
print(result.group(1))
print(result.group(2))
```

## 3、反向引用（后向引用）

在正则表达式中，我们可以通过\n（n代表第n个缓存区的编号）来引用缓存区中的内容，我们把这个过程就称之为"反向引用"。

① 连续4个数字

re.search(r'\d\d\d\d, str1)

1234、5678、6789

② 连续的4个数字，但是数字的格式为1111、2222、3333、4444、5555效果？

 re.search(r'(\d)\1\1\1, str1)

## 4、几个练习题

① 查找连续的四个数字，如：3569

答：`\d\d\d\d或\d{4}`

② 查找连续的相同的四个数字，如：1111
答：`(\d)\1\1\1`

③ 查找数字，如：1221,3443
答：`(\d)(\d)\2\1`

第一个()放入1号缓冲区，如果想引用\1

第二个()放入2号缓冲区，如果想引用\2

④ 查找字符，如：AABB,TTMM（提示：A-Z，正则：[A-Z]）
答：`([A-Z])\1([A-Z])\2`

⑤ 查找连续相同的四个数字或四个字符（提示：\w）

答：`(\w)\1\1\1`

1111

aaaa

bbbb

# 五、正则表达式其他方法

## 1、选择匹配符

`|`可以匹配多个规则

案例：匹配字符串hellojava或hellopython

```python
import re


str = 'hellojava, hellopython'
result = re.finditer(r'hello(java|python)', str)
if result:
    for i in result:
        print(i.group())
else:
    print('未匹配到任何数据')
```

## 2、分组别名

| 代码       | 功能                             |
| ---------- | -------------------------------- |
| (?P<name>) | 分组起别名                       |
| (?P=name)  | 引用别名为name分组匹配到的字符串 |

案例：匹配\<book>\</book>

```python
# 导入模块
import re

str1 = '<book></book>'
result = re.search(r'<(?P<mark>\w+)></(?P=mark)>', str1)

print(result.group())
```

## 3、综合案例

①需求：在列表中["apple", "banana", "orange", "pear"]，匹配apple和pear

```python
import re

list1 = ["apple", "banana", "orange", "pear"]
str1 = str(list1)
result = re.finditer('(apple|pear)', str1)
if result:
    for i in result:
        print(i.group())
else:
    print('未匹配到任何数据')
```

② 需求：匹配出163、126、qq等邮箱

```python
import re

email = '1478670@qq.com, go@126.com, heima123@163.com'
result = re.finditer('\w+@(qq|126|163).com', email)
if result:
    for i in result:
        print(i.group())
else:
    print('未匹配到任何数据')
```

③需求 :  匹配qq:10567这样的数据，提取出来qq文字和qq号码

```python
import re

str1 = 'qq:10567'
result = re.split(r':', str1)
if result:
    print(f'{result[0]}号：{result[1]}')
else:
    print('未匹配到任何数据')
```

④需求：匹配出<html>hh</html>

⑤需求：匹配出<html><h1>www.itcast.cn</h1></html>

⑥需求：匹配出<html><h1>www.itcast.cn</h1></html>

