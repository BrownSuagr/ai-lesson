import torch.nn as nn
import torch.nn.functional as F
from dm_decoder import *
# 定义输出层
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # d_model代表词嵌入维度，vocab_size：代表最后的输出维度：词表的大小
        # 定义全连接层
        self.linear = nn.Linear(d_model, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        y = F.log_softmax(self.linear(x), dim=-1)
        return y

# 测试输出
def test_output():
    # 得到解码器的输出结果
    decoder_output = test_decoder()
    # 实例化输出层
    my_generator = Generator(d_model=512, vocab_size=1000)
    result = my_generator(decoder_output)
    print(f'result最终transformer结果的输出-->{result.shape}')
    print(f'result最终transformer结果的输出-->{result}')

if __name__ == '__main__':
    test_output()

