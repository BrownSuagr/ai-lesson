### 学习目标

* 了解加载和使用预训练模型的工具.
* 掌握加载和使用预训练模型的过程.



## 1 加载和使用预训练模型的工具

* 在这里我们使用Transformers工具包进行模型的加载和使用.
* 这些预训练模型由世界先进的NLP研发团队huggingface提供.
* **注意: 下面使用的代码需要国外服务器的资源, 在国内使用的时候, 国内的网站下载可能会出现在原地卡死不动, 或是网络连接超时等一些网络报错, 均是网络问题, 不是代码问题, 这个可以先行跳过, 把主要逻辑梳理完成即可**



## 2 加载和使用预训练模型的步骤

* 第一步: 确定需要加载的预训练模型并安装依赖包.
* 第二步: 加载预训练模型的映射器tokenizer.
* 第三步: 加载带/不带头的预训练模型.
* 第四步: 使用模型获得输出结果.



### 2.1 确定需要加载的预训练模型并安装依赖包

* 能够加载哪些模型可以参考前一小结中的常用预训练模型
* 这里假设我们处理的是中文文本任务, 需要加载的模型是BERT的中文模型: bert-base-chinese
* 在使用工具加载模型前需要安装必备的依赖包:

```shell
pip install tqdm boto3 requests regex sentencepiece sacremoses
```




### 2.2 加载预训练模型的映射器tokenizer

```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

mirror='https://mirrors.tuna.tsinghua.edu.cn/help/hugging-face-models/'

def demo24_1_load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",mirror='https://mirrors.tuna.tsinghua.edu.cn/help/hugging-face-models/')
    print("tokenizer--->", tokenizer)
    
demo24_1_load_tokenizer()
```




### 2.3 加载带/不带头的预训练模型

* 加载预训练模型时我们可以选择带头或者不带头的模型
* 这里的'头'是指模型的任务输出层, 选择加载不带头的模型, 相当于使用模型对输入文本进行特征表示.
* 选择加载带头的模型时, 有三种类型的'头'可供选择,  AutoModelForMaskedLM (语言模型头), AutoModelForSequenceClassification (分类模型头),  AutoModelForQuestionAnswering (问答模型头)
* 不同类型的'头', 可以使预训练模型输出指定的张量维度. 如使用'分类模型头', 则输出尺寸为(1,2)的张量, 用于进行分类任务判定结果.

```python
# 加载不带头的预训练模型
def demo24_2_load_model():

    # 加载的预训练模型的名字
    model_name = 'bert-base-chinese'

    print('加载不带头的预训练模型')
    model =AutoModel.from_pretrained(model_name)
    print('model--->', model)


    # 加载带有语言模型头的预训练模型
    print('加载带有语言模型头的预训练模型')
    lm_model =AutoModelForMaskedLM.from_pretrained(model_name)
    print('lm_model--->', lm_model)

    # 加载带有分类模型头的预训练模型
    print('加载带有分类模型头的预训练模型')
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print('classification_model--->', classification_model)

    # 加载带有问答模型头的预训练模型
    print('加载带有问答模型头的预训练模型')
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print('qa_model--->', qa_model)

demo24_2_load_model()
```



### 2.4 使用模型获得输出结果

#### 1 使用不带头的模型进行输出

```python
def demo24_3_load_AutoModel():

    # 加载的预训练模型的名字
    model_name = 'bert-base-chinese'

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",mirror=mirror)

    # 2 加载model
    model = AutoModel.from_pretrained(model_name)

    # 3 使用tokenizer 文本数值化
    # 输入的中文文本
    input_text = "人生该如何起头"

    # 使用tokenizer进行数值映射
    indexed_tokens = tokenizer.encode(input_text)

    # 打印映射后的结构
    print("indexed_tokens:", indexed_tokens)

    # 将映射结构转化为张量输送给不带头的预训练模型
    tokens_tensor = torch.tensor([indexed_tokens])

    # 4 使用不带头的预训练模型获得结果
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, return_dict=False)
        # encoded_layers, _ = model(tokens_tensor)

    print("不带头的模型输出结果:", encoded_layers)
    print("不带头的模型输出结果的尺寸:", encoded_layers.shape)
    
demo24_3_load_AutoModel()
```



> * 输出效果:

```python
# tokenizer映射后的结果, 101和102是起止符, 
# 中间的每个数字对应"人生该如何起头"的每个字.
indexed_tokens: [101, 782, 4495, 6421, 1963, 862, 6629, 1928, 102]


不带头的模型输出结果: tensor([[[ 0.5421,  0.4526, -0.0179,  ...,  1.0447, -0.1140,  0.0068],
         [-0.1343,  0.2785,  0.1602,  ..., -0.0345, -0.1646, -0.2186],
         [ 0.9960, -0.5121, -0.6229,  ...,  1.4173,  0.5533, -0.2681],
         ...,
         [ 0.0115,  0.2150, -0.0163,  ...,  0.6445,  0.2452, -0.3749],
         [ 0.8649,  0.4337, -0.1867,  ...,  0.7397, -0.2636,  0.2144],
         [-0.6207,  0.1668,  0.1561,  ...,  1.1218, -0.0985, -0.0937]]])


# 输出尺寸为1x9x768, 即每个字已经使用768维的向量进行了表示,
# 我们可以基于此编码结果进行接下来的自定义操作, 如: 编写自己的微调网络进行最终输出.
不带头的模型输出结果的尺寸: torch.Size([1, 9, 768])
```



#### 2 使用带有语言模型头的模型进行输出

```python
def demo24_4_load_AutoLM():

    # 1 加载 tokenizer
    # 加载的预训练模型的名字
    model_name = 'bert-base-chinese'

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",mirror=mirror)

    # 2 加载model
    lm_model =AutoModelForMaskedLM.from_pretrained(model_name)

    # 3 使用tokenizer 文本数值化
    # 输入的中文文本
    input_text = "人生该如何起头"

    # 使用tokenizer进行数值映射
    indexed_tokens = tokenizer.encode(input_text)

    # 打印映射后的结构
    print("indexed_tokens:", indexed_tokens)

    # 将映射结构转化为张量输送给不带头的预训练模型
    tokens_tensor = torch.tensor([indexed_tokens])

    # 使用带有语言模型头的预训练模型获得结果
    with torch.no_grad():
        lm_output = lm_model(tokens_tensor,return_dict=False)

    print("带语言模型头的模型输出结果:", lm_output)
    print("带语言模型头的模型输出结果的尺寸:", lm_output[0].shape)

demo24_4_load_AutoLM()
```



> * 输出效果:

```text
带语言模型头的模型输出结果: (tensor([[[ -7.9706,  -7.9119,  -7.9317,  ...,  -7.2174,  -7.0263,  -7.3746],
         [ -8.2097,  -8.1810,  -8.0645,  ...,  -7.2349,  -6.9283,  -6.9856],
         [-13.7458, -13.5978, -12.6076,  ...,  -7.6817,  -9.5642, -11.9928],
         ...,
         [ -9.0928,  -8.6857,  -8.4648,  ...,  -8.2368,  -7.5684, -10.2419],
         [ -8.9458,  -8.5784,  -8.6325,  ...,  -7.0547,  -5.3288,  -7.8077],
         [ -8.4154,  -8.5217,  -8.5379,  ...,  -6.7102,  -5.9782,  -7.6909]]]),)

# 输出尺寸为1x9x21128, 即每个字已经使用21128维的向量进行了表示, 
# 同不带头的模型一样, 我们可以基于此编码结果进行接下来的自定义操作, 如: 编写自己的微调网络进行最终输出.
带语言模型头的模型输出结果的尺寸: torch.Size([1, 9, 21128])
```



#### 3 使用带有分类模型头的模型进行输出

```python
def demo24_5_load_AutoSeqC():

    # 1 加载 tokenizer
    # 加载的预训练模型的名字
    model_name = 'bert-base-chinese'

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",mirror=mirror)

    # 2 加载model
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 3 使用tokenizer 文本数值化
    # 输入的中文文本
    input_text = "人生该如何起头"

    # 使用tokenizer进行数值映射
    indexed_tokens = tokenizer.encode(input_text)

    # 打印映射后的结构
    print("indexed_tokens:", indexed_tokens)

    # 将映射结构转化为张量输送给不带头的预训练模型
    tokens_tensor = torch.tensor([indexed_tokens])

    # 使用带有分类模型头的预训练模型获得结果
    with torch.no_grad():
        classification_output = classification_model(tokens_tensor)

    print("带分类模型头的模型输出结果:", classification_output)
    print("带分类模型头的模型输出结果的尺寸:", classification_output[0].shape)
    
demo24_5_load_AutoSeqC()
```



> * 输出效果:

```text
带分类模型头的模型输出结果: (tensor([[-0.0649, -0.1593]]),)
# 输出尺寸为1x2, 可直接用于文本二分问题的输出
带分类模型头的模型输出结果的尺寸: torch.Size([1, 2])
```



#### 4 使用带有问答模型头的模型进行输出

```python
def demo24_6_load_AutoQA():

    # 1 加载 tokenizer
    # 加载的预训练模型的名字
    model_name = 'bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",mirror=mirror)

    # 2 加载model
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # 3 使用
    # 使用带有问答模型头的模型进行输出时, 需要使输入的形式为句子对
    # 第一条句子是对客观事物的陈述
    # 第二条句子是针对第一条句子提出的问题
    # 问答模型最终将得到两个张量,
    # 每个张量中最大值对应索引的分别代表答案的在文本中的起始位置和终止位置
    input_text1 = "我家的小狗是黑色的"
    input_text2 = "我家的小狗是什么颜色的呢?"

    # 映射两个句子
    indexed_tokens = tokenizer.encode(input_text1, input_text2)
    print("句子对的indexed_tokens:", indexed_tokens)

    # 输出结果: [101, 2769, 2157, 4638, 2207, 4318, 3221, 7946, 5682, 4638, 102, 2769, 2157, 4638, 2207, 4318, 3221, 784, 720, 7582, 5682, 4638, 1450, 136, 102]
    #
    # 用0，1来区分第一条和第二条句子
    segments_ids = [0] * 11 + [1] * 14

    # 转化张量形式
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])

    # 使用带有问答模型头的预训练模型获得结果
    with torch.no_grad():
        start_logits, end_logits = qa_model(tokens_tensor, token_type_ids=segments_tensors, return_dict=False)

    print("带问答模型头的模型输出结果:", (start_logits, end_logits))
    print("带问答模型头的模型输出结果的尺寸:", (start_logits.shape, end_logits.shape))  # (torch.Size([1, 25]), torch.Size([1, 25]))
    
demo24_6_load_AutoQA()
```



> * 输出效果:

```text
句子对的indexed_tokens: [101, 2769, 2157, 4638, 2207, 4318, 3221, 7946, 5682, 4638, 102, 2769, 2157, 4638, 2207, 4318, 3221, 784, 720, 7582, 5682, 4638, 1450, 136, 102]

带问答模型头的模型输出结果: (tensor([[ 0.2574, -0.0293, -0.8337, -0.5135, -0.3645, -0.2216, -0.1625, -0.2768,
         -0.8368, -0.2581,  0.0131, -0.1736, -0.5908, -0.4104, -0.2155, -0.0307,
         -0.1639, -0.2691, -0.4640, -0.1696, -0.4943, -0.0976, -0.6693,  0.2426,
          0.0131]]), tensor([[-0.3788, -0.2393, -0.5264, -0.4911, -0.7277, -0.5425, -0.6280, -0.9800,
         -0.6109, -0.2379, -0.0042, -0.2309, -0.4894, -0.5438, -0.6717, -0.5371,
         -0.1701,  0.0826,  0.1411, -0.1180, -0.4732, -0.1541,  0.2543,  0.2163,
         -0.0042]]))


# 输出为两个形状1x25的张量, 他们是两条句子合并长度的概率分布,
# 第一个张量中最大值所在的索引代表答案出现的起始索引, 
# 第二个张量中最大值所在的索引代表答案出现的终止索引.
带问答模型头的模型输出结果的尺寸: (torch.Size([1, 25]), torch.Size([1, 25]))
```



## 3 小结

* 加载和使用预训练模型的工具:

    * 在这里我们使用transformers工具进行模型的加载和使用.
    * 这些预训练模型由世界先进的NLP研发团队huggingface提供.

* 加载和使用预训练模型的步骤:

    * 第一步: 确定需要加载的预训练模型并安装依赖包.
    * 第二步: 加载预训练模型的映射器tokenizer.
    * 第三步: 加载带/不带头的预训练模型.
    * 第四步: 使用模型获得输出结果.

