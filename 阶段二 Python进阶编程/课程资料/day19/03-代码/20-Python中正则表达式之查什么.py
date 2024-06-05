# 导入模块
import re

# # 定义一个字符串
# str1 = '&'
# # 使用正则匹配任意某个字符（不确定这个字符是什么）
# result = re.match('.', str1)
# # 输出匹配结果
# print(result.group())

# 定义一个字符串
# str2 = '9abcdefg'
# result = re.match('[0-9]', str2)
# print(result.group())

# 定义一个字符串
# str3 = '$123456'
# result = re.match('[^0-9]', str3)
# print(result.group())

# 定义一个字符串
# str4 = '8abcdefg'
# result = re.match('\d', str4)
# print(result.group())

# 定义一个字符串
# str5 = 'abcdefg7'
# result = re.match('\D', str5)
# print(result.group())

# 定义一个字符串
# str6 = ' abcdefg'
# result = re.match('\s', str6)
# print(len(result.group()))

# str6 = 'abcdefg'
# result = re.match('\S', str6)
# print(result.group())

# str7 = 'admin@itcast.cn'
# result = re.match('\w', str7)
# print(result.group())

str8 = '$admin@itcast.cn'
result = re.match('\W', str8)
print(result.group())