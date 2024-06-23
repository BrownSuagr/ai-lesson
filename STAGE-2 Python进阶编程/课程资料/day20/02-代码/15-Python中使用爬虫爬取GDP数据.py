'''
爬取页面：http://127.0.0.1:8000/gdp.html
获取数据的最终格式：[['美国', 218463.3], ['中国', 155518.9], ['日本', 52797.7]]

<a href=""><font>美国</font></a>
<font>$218463.3<font>亿</font></font>
'''
# 导入模块
import requests
import re

# 定义一个全局变量
country = []  # 国家
country_data = []  # 国家的GDP数据

# 定义一个函数，专门用于获取GDP数据
def get_gdp_data():
    global country
    global country_data

    data = requests.get('http://127.0.0.1:8000/gdp.html')
    # 获取html数据
    data = data.content.decode('utf-8')
    # 定义一个列表，对\n换行符进行切割
    data_list = data.split('\n')
    # 对data_list列表进行遍历操作，每遍历1次得到一行
    for line in data_list:
        result = re.match('.*<a href=""><font>(.*)</font></a>', line)
        if result:
            country.append(result.group(1))
        # print(line)
        result = re.match('.*<font>\$(.*)<font>亿', line)
        if result:
            country_data.append(result.group(1))

    data = list(zip(country, country_data))
    print(data)

# 定义程序的执行入口
if __name__ == '__main__':
    get_gdp_data()