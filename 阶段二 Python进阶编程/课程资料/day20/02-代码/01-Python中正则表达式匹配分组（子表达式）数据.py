# 导入模块
import re

# 定义一个字符串，匹配字符串中的123，然后能单独获取2和3
str1 = 'abcdef123ghijklmn'
# 如果正则中带有分组（子表达式）数据，建议使用search方法或finditer方法获取数据
result = re.search(r'\d(\d)(\d)', str1)
# 以上正则匹配完成后，会产生3个结果，第一个结果result.group()默认获取整个正则匹配的结果
print(result.group())
# 第二个结果，使用result.group(缓冲区号码 => 1)
print(result.group(1))
# 第三个结果，使用result.group(缓冲区号码 => 2）
print(result.group(2))