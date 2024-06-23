'''
需求：在列表中["apple", "banana", "orange", "pear"]，匹配apple和pear
'''
import re

list1 = ["Apple", "banana", "orange", "pear"]
# 将其转换为字符串类型
str1 = str(list1)
# print(str1)
# print(type(str1))
# 使用选择匹配符匹配apple和pear
result = re.finditer(r'(apple|pear)', str1, re.I)
for i in result:
    print(i.group())