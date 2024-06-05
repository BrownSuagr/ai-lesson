# 导入模块
import re

# 定义一个字符串，字符串中保存11位手机号码，验证手机号码是否合理
mobile = '13575008994'
# 定义一个正则表达式，验证手机号码是否合理
result = re.match('^1[3456789]\d{9}$', mobile)
# 判断是否合理
if result:
    print('合理的手机号码')
else:
    print('不合理的手机号码')