# 1、导入模块
# 导入FastAPI类
from fastapi import FastAPI
# 导入Response类，用于返回数据给浏览器端
from fastapi import Response
# 导入uvicorn服务器模块，用于保持此文件一直运行（类似while True）
import uvicorn

# 2、创建FastAPI对象
app = FastAPI()

# 3、通过@app路由装饰器收发信息
# 路由 浏览器请求 => 服务器响应对应关系
# 接收浏览器发送过来的请求
@app.get('/gdp.html')
# 响应数据给浏览器端
def main():
    # 读取要返回的文件内容
    with open('source/html/gdp.html', 'rb') as f:
        data = f.read()

    # 把data数据以text/html格式返回给浏览器端
    return Response(content=data, media_type='text/html')

# 4、运行服务器
# FastAPI对象
# 绑定IP
# 绑定端口
uvicorn.run(app, host="192.168.27.93", port=8000)