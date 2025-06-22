# coding:utf-8
from transformers import pipeline
import numpy as np
# from rich import print
# from pprint import pprint

# todo: 1.实现文本分类任务
def dm01_text_classfication():
    # 1.直接调用pipeline方法,获得模型对象
    # model = pipeline(task='sentiment-analysis', model='./model/chinese_sentiment')
    model = pipeline(task='text-classification', model='./model/chinese_sentiment')
    # 2.根据model去预测
    result = model("我爱北京天安门，天安门上太阳升。")
    print(f'文本分类结果-->{result}')


# todo: 2.实现特征抽取任务
def dm02_feature_extraction():
    # 1.直接调用pipeline方法,获得模型对象
    model = pipeline(task='feature-extraction', model='./model/bert-base-chinese')
    # 2.根据model去预测
    result = model("人生该如何起头")
    print(f'特征抽取结果-->{np.array(result).shape}')


# todo: 3.实现完形填空任务
def dm03_fill_mask():
    # 1.直接调用pipeline方法,获得模型对象
    model = pipeline(task='fill-mask', model='./model/chinese-bert-wwm')
    # 2.根据model去预测
    result = model("我想去[MASK]家吃饭")
    print(f'完形填空结果-->{result}')


# todo: 4.实现阅读理解任务
def dm04_qa():
    # 1.直接调用pipeline方法,获得模型对象
    model = pipeline(task='question-answering', model='./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 2.准备数据
    context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    # 3.根据model去预测
    result = model(context=context, question=questions)
    print(f'阅读理解任务结果-->{result}')


# todo: 5.实现文本摘要任务
def dm05_summary():
    # 1.直接调用pipeline方法,获得模型对象
    model = pipeline(task='summarization', model='./model/distilbart-cnn-12-6')
    # 2.准备数据
    context = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."
    # 3.根据model去预测
    result = model(context)
    print(f'文本摘要任务结果-->{result}')


# todo: 6.实现NER任务
def dm06_ner():
    # 1.直接调用pipeline方法,获得模型对象
    model = pipeline(task='token-classification', model='./model/roberta-base-finetuned-cluener2020-chinese')
    # 2.准备数据
    # 3.根据model去预测
    # result = model("我爱北京天安门，天安门上太阳升。")
    result = model("我在广州上课，这里很热")
    print(f'NER任务结果-->{result}')
if __name__ == '__main__':
    # dm01_text_classfication()
    # dm02_feature_extraction()
    # dm03_fill_mask()
    # dm04_qa()
    # dm05_summary()
    dm06_ner()