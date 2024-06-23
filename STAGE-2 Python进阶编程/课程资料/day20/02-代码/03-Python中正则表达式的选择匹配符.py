'''
案例：匹配字符串hellojava或hellopython
'''
# 导入模块
import re

# 定义一个字符串
str1 = 'hellojava, hellopython'
# 编写正则表达式匹配hellojava或hellopython
result = re.finditer(r'hello(java|python)', str1)
if result:
    # 匹配到了结果
    for i in result:
        print(i.group())
else:
    # 没有匹配到任何结果
    print('暂未匹配到任何内容')