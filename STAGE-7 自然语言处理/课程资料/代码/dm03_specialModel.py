# coding:utf-8
from transformers import BertTokenizer, BertForMaskedLM
import torch
# todo: 3.实现完形填空任务
def dm03_fill_mask():
    # 1.加载tokenizer分词器
    fm_tokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese')
    # 2. 加载model
    fm_model = BertForMaskedLM.from_pretrained('./model/bert-base-chinese')
    # 3. 准备数据
    message = "我想明天去[MASK]家吃饭."
    # 3.1 对上述字符串message进行编码，变成向量送给模型
    inputs = fm_tokenizer.encode_plus(message, return_tensors='pt')
    print(f'inputs--》{inputs}')
    # 4.设置模型为评估模式
    fm_model.eval()
    # # 5. 将数据送入model
    result = fm_model(**inputs)
    print(f'result结果--》{result}')
    # logits--》[1,12,21128]
    logits = result["logits"]
    print(f'logits--》{logits.shape}')
    # 取出[MASK]位置，对应预测最大概率的索引值
    idx = torch.argmax(logits[0, 6]).item()
    token = fm_tokenizer.convert_ids_to_tokens([idx])
    print(f'完形填空最后MASK位置预测的结果为{token}')

if __name__ == '__main__':
    dm03_fill_mask()
