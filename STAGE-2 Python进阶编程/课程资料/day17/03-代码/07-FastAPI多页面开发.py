# 1、导入模块
from fastapi import FastAPI
from fastapi import Response
import uvicorn

# 2、创建FastAPI对象
app = FastAPI()

# 3、使用路由装饰器收发信息
@app.get('/index.html')
def index():
    with open('source/html/index.html', 'rb') as f:
        data = f.read()

    # 返回数据给客户端浏览器
    return Response(content=data, media_type='text/html')

@app.get('/gdp.html')
def gdp():
    with open('source/html/gdp.html', 'rb') as f:
        data = f.read()

    # 返回数据给客户端浏览器
    return Response(content=data, media_type='text/html')

# 4、创建服务器，让程序一直运行
uvicorn.run(app, host='192.168.27.93', port=8000)