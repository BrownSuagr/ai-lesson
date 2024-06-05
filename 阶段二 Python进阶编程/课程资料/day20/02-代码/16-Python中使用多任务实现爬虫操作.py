'''
1、爬取站点的html页面，获取html代码
2、对html代码进行解析，找到图片地址，获取过来
3、根据请求的url图片连接，使用open方法将其保存到本地
'''
import requests
import re
import threading

# 定义一个全局变量
country = []  # 国家
country_data = []  # 国家的GDP数据

def save_pic(url_list):
    num = 0
    for picurl in url_list:
        # 每循环次1次，模拟发送请求，获取图片字节流数据
        data = requests.get(picurl)
        data = data.content  # 不要解码，否则无法保存资源图片
        with open(f'source/spyder/{num}.jpg', 'wb') as f:
            f.write(data)
        num += 1


def get_pic():
    # 爬取http://127.0.0.1:8000/index.html页面的所有信息
    data = requests.get('http://127.0.0.1:8000/index.html')
    # 爬取到的内容，要根据.content属性获取数据
    data = data.content.decode('utf-8')
    # print(data)
    # 对data数据进行解析，采集所有的.jpg图片
    data_list = data.split('\n')
    # 使用for循环对data_list列表进行遍历
    # <td><img src="./images/1.jpg" width="184px" height="122px" /></td>
    url_list = []
    for line in data_list:
        # 对line变量进行解析，获取图片标签中的src图片地址
        result = re.match('.*src="(.*)" width.*', line)
        if result:
            url = result.group(1)  # ./images/0.jpg => http://127.0.0.1:8000/images/0.jpg
            url = 'http://127.0.0.1:8000' + url[1:]
            url_list.append(url)
    return url_list

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
    return data

if __name__ == '__main__':
    get_pic_thread = threading.Thread(target=get_pic)
    get_gdp_thread = threading.Thread(target=get_gdp_data)

    get_pic_thread.start()
    get_gdp_thread.start()
