# 导入模块
import re

# 定义一个字符串，匹配连续的3位数字
# str1 = '123abc'
# result = re.match('\d{3}', str1)
# print(result.group())

# 定义一个字符串，匹配多个字符（最少匹配0个，最多匹配n个）
# str2 = '20211128abc.jpg'
# result = re.match('.*', str2)
# print(result.group())

# 定义一个字符串，匹配1或10
str3 = '10'
result = re.match('10?', str3)
print(result.group())