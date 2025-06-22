import requests
import json


def request_third_part_llm(model_name: str, content_json: json):
    '''
    请求第三方模型集成平台
    :param model_name:
    :param content_json:
    :return:
    '''
    url = "https://www.dmxapi.cn/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        'Authorization': 'sk-D5PwNWjoJrBlulzHXstu37ps2AQbJkp8zYhgQmaHkvE58zOc',  # 这里放你的 DMXapi key
        'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.cn)',
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
        # "model": "qwen2.5-72b-instruct",
        # 这里是你需要访问的模型，改成上面你需要测试的模型名称就可以了。
        "model": model_name,
        "messages": content_json
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

from langchain_openai import ChatOpenAI
def request_third_part_langChain(model_name: str, content_json: json):

    model = ChatOpenAI(
        model="gpt-4o-mini",
        base_url="https://www.dmxapi.cn/v1",
        api_key="sk-***************************************************",  # 替换成你的 DMXapi 令牌key
    )
    text = "周树人和鲁迅是兄弟吗？"
    print(model(text))


if __name__ == '__main__':
    msg = [{"role": "user", "content": "周树人和鲁迅是兄弟吗？"}]
    result = request_third_part_llm('qwen2.5-72b-instruct', msg)
    print(result)
