# 导入工具包
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering
# AutoModelForSeq2SeqLM：文本摘要
# AutoModelForTokenClassification：ner
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# todo: 1.实现文本分类任务
def dm01_text_classfication():
    # 1.加载tokenizer分词器
    cls_tokenizer = AutoTokenizer.from_pretrained('./model/chinese_sentiment')
    # 2. 加载model
    cls_model = AutoModelForSequenceClassification.from_pretrained('./model/chinese_sentiment')
    # print(cls_model)
    # 3. 准备数据
    message = "我爱北京天安门"
    # 3.1 对上述字符串message进行编码，变成向量送给模型
    # return_tensors='pt'这里代表返回张量；padding='max_length',不足最大句子长度的补齐； truncation=True超过最大句子长度的进行截断
    inputs = cls_tokenizer.encode(message, return_tensors='pt', padding='max_length', truncation=True, max_length=20)
    # inputs = cls_tokenizer.encode_plus(message, return_tensors='pt', padding='max_length', truncation=True, max_length=20)
    print(f'inputs--》{inputs}')
    # inputs = cls_tokenizer.encode(message, padding='max_length', truncation=True, max_length=20)
    # inputs = torch.tensor([inputs], dtype=torch.long)
    # print(f'inputs--》{inputs}')
    # 4.设置模型为评估模式
    cls_model.eval()
    # 5. 将数据送入model
    result = cls_model(inputs)
    # result = cls_model(**inputs)
    print(f'result文本分类的结果--》{result}')
    # # logits = result["logits"]
    # logits = result.logits
    # print(f'logits--》{logits}')
    # topv, topi = torch.topk(logits, k=1)
    # print(f'topv=-》{topv}')
    # print(f'topi=-》{topi}')


# todo: 2.实现特征抽取任务
def dm02_feature_extraction():
    # 1.加载tokenizer分词器
    fe_tokenizer = AutoTokenizer.from_pretrained('./model/bert-base-chinese')
    # 2. 加载model
    fe_model = AutoModel.from_pretrained('./model/bert-base-chinese')
    # print(cls_model)
    # 3. 准备数据
    message = ['你是谁', '人生该如何起头']
    # 3.1 对上述字符串message进行编码，变成向量送给模型
    # return_tensors='pt'这里代表返回张量；padding='max_length',不足最大句子长度的补齐； truncation=True超过最大句子长度的进行截断
    inputs = fe_tokenizer.encode_plus(message, return_tensors='pt', padding='max_length', truncation=True, max_length=30)
    print(f'inputs--》{inputs}')
    # 4.设置模型为评估模式
    fe_model.eval()
    # 5. 将数据送入model
    # result = fe_model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"], attention_mask=inputs["attention_mask"])
    result = fe_model(**inputs)
    print(f'result特征抽取的结果--》{result}')
    print(f'last_hidden_state-->{result["last_hidden_state"].shape}')
    print(f'pooler_output-->{result["pooler_output"].shape}')


# todo: 3.实现完形填空任务
def dm03_fill_mask():
    # 1.加载tokenizer分词器
    fm_tokenizer = AutoTokenizer.from_pretrained('./model/chinese-bert-wwm')
    # 2. 加载model
    fm_model = AutoModelForMaskedLM.from_pretrained('./model/chinese-bert-wwm')
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

# todo: 4.实现阅读理解任务
def dm04_qa():
    # 1.加载tokenizer分词器
    qa_tokenizer = AutoTokenizer.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 2. 加载model
    qa_model = AutoModelForQuestionAnswering.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 3. 准备数据
    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    # 4.设置模型为评估模式
    qa_model.eval()
    # # 5. 将数据送入model
    for question in questions:
        # 对数据进行encode
        inputs = qa_tokenizer.encode_plus(question, context, return_tensors='pt')
        # print(f'inputs--》{inputs}')
        # 将inputs送入模型
        result = qa_model(**inputs)
        start_logits = result["start_logits"]
        end_logits = result["end_logits"]
        start_idx = torch.argmax(start_logits)
        # print(f'start_idx--》{start_idx}')
        end_idx = torch.argmax(end_logits) + 1
        # print(f'end_idx--》{end_idx}')
        # 获取对应的答案
        answer = qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx: end_idx])
        print(f'当前的问题是：{question}, 对应的答案是：{answer}')

# todo: 5.实现文本摘要任务
def dm05_summary():
    # 1.加载tokenizer分词器
    summary_tokenizer = AutoTokenizer.from_pretrained('./model/distilbart-cnn-12-6')
    # 2. 加载model
    summary_model = AutoModelForSeq2SeqLM.from_pretrained('./model/distilbart-cnn-12-6')
    # 3. 准备数据
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
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
    # 3.1 对上述字符串message进行编码，变成向量送给模型
    # inputs = summary_tokenizer(text, return_tensors='pt')
    inputs = summary_tokenizer.encode_plus(text, return_tensors='pt')
    # print(f'摘要任务inputs-->{inputs}')
    # print(f'摘要任务inputs-->{len(inputs)}')
    # 4.将数据送给模型
    summary_model.eval()
    # output = summary_model.generate(**inputs)
    output = summary_model.generate(inputs["input_ids"])
    # print(f'output-->{output}')
    print(f'output-->{output.shape}')

    # print(summary_tokenizer.convert_ids_to_tokens(output[0])) # 不行
    # skip_special_tokens跳过特殊字符；clean_up_tokenization_spaces将标点符号和单词分开
    result = summary_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f'摘要的最终结果--》{result}')


# todo: 6.实现NER任务
def dm06_ner():
    # 1.加载tokenizer分词器
    ner_tokenizer = AutoTokenizer.from_pretrained('./model/roberta-base-finetuned-cluener2020-chinese')
    # 2. 加载model
    ner_model = AutoModelForTokenClassification.from_pretrained('./model/roberta-base-finetuned-cluener2020-chinese')
    # 3.加载config配置文件
    ner_config = AutoConfig.from_pretrained('./model/roberta-base-finetuned-cluener2020-chinese')
    # print(f'ner_config-->{ner_config}')
    # 4.准备数据
    message = "我爱北京天安门，天安门上太阳升"
    # 数据编码
    inputs = ner_tokenizer.encode_plus(message, return_tensors='pt')
    print(f'inputs--->{inputs}')

    # 5.将数据送入模型
    ner_model.eval()
    result = ner_model(inputs["input_ids"])
    # result = ner_model(**inputs)
    logits = result["logits"]
    print(f'logits--》{logits.shape}')
    # 得到原始的句子
    input_tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f'input_tokens--》{input_tokens}')
    output = []
    # 对每个单词找出对应的预测标签

    for token, value in zip(input_tokens, logits[0]):
        if token in ner_tokenizer.all_special_tokens:
            continue
        # print(f'token-->{token}')
        # print(f'value-->{value}')
        idx = torch.argmax(value).item()
        # print(idx)
        label = ner_config.id2label[idx]
        # print(f'label--》{label}')
        output.append([token, label])
    print(output)

if __name__ == '__main__':
    # dm01_text_classfication()
    # dm02_feature_extraction()
    # dm03_fill_mask()
    # dm04_qa()
    # dm05_summary()
    dm06_ner()