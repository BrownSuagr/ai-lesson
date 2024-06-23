# 导入模块
import re

# 定义一个字符串，主要是邮箱信息
str1 = '1478670@qq.com, go@126.com, heima123@163.com'

# 定义一个正则表达式，匹配163、126、qq等邮箱
result = re.finditer('\w+@(qq|126|163)\.com', str1)

if result:
    for i in result:
        print(i.group())
else:
    print('暂未匹配到任何相关内容')