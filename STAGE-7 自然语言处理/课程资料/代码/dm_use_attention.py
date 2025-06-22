# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# 按照第一种计算规则实现注意力
'''
计算注意力的步骤：
1。按照注意力计算规则，实现Q、K、V的计算
2。如果第一步，按照拼接的方式实现的计算，那么需要将Q和第一步计算的结果再次拼接；否则，不用
3。如果第二步Q和第一步计算的结果已经拼接，那么需要进行再次线性变化，得到最终的输出结果（指定输出维度）；但是如果没有拼接，就不用线性变化

如果第一步按照不拼接的方式进行，就不用再进行第2和3步。
'''
# 维度需要变化（降和升）
class MyAtten(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # query_size代表query输入维度（最后一维）
        self.query_size = query_size
        # key_size代表key输入维度（最后一维）
        self.key_size = key_size
        # value_size1代表value输入第二维度
        self.value_size1 = value_size1
        # value_size2代表value输入第三维度（最后一维）
        self.value_size2 = value_size2
        # output_size代表指定的输出维度
        self.output_size = output_size
        # 定义第一个全连接层：计算权重分数的
        # value_size1作为输出维度，是为了和Value进行三维矩阵想乘
        self.atten = nn.Linear(query_size+key_size, value_size1)
        # 定义第二个全连接层：让结果按照指定维度输出
        # 按照注意力计算步骤的第2步：如果第一步，按照拼接的方式实现的计算，那么需要将Q和第一步计算的结果再次拼接
        # 第3步：如果第二步Q和第一步计算的结果已经拼接，那么需要进行再次线性变化，得到最终的输出结果（指定输出维度）
        self.attn_combin = nn.Linear(query_size+value_size2, output_size)

    def forward(self, Q, K, V):
        # Q--》[1,1,32]; K--》[1,1,32];V--》[1, 32, 64]
        # 第一步：计算注意力权重分数
        # Q[0]-->[1, 32];K[0]-->[1, 32]-->torch.cat((Q[0], K[0]))-->[1,64]
        # atten_weight -->[1, 32]
        atten_weight = F.softmax(self.atten(torch.cat((Q[0], K[0]), dim=-1)), dim=-1)
        # atten_weigght--》升维--[1,1,32]要和V[1, 32, 64]进行想乘;atten1代表未更新的注意力--》[1,1,64]
        # 第二步：完成初步注意力的计算
        atten1 = torch.bmm(atten_weight.unsqueeze(dim=0), V)
        # 第三步：将Q和第一步计算的结果再次拼接,再经过线性变化得到指定输出维度
        # Q[0]--->[1, 32];  atten1[0]-->[1, 64]--》拼接后[1，96]
        # ouput-->[1, 32]
        output = self.attn_combin(torch.cat((Q[0], atten1[0]), dim=-1))
        # 把output升维度;result-->【1，1，32】
        result = output.unsqueeze(dim=0)
        return result, atten_weight


# 自定义和，讲义有点区别的注意力计算

class GzMyAtten(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # query_size代表query输入维度（最后一维）
        self.query_size = query_size
        # key_size代表key输入维度（最后一维）
        self.key_size = key_size
        # value_size1代表value输入第二维度
        self.value_size1 = value_size1
        # value_size2代表value输入第三维度（最后一维）
        self.value_size2 = value_size2
        # output_size代表指定的输出维度
        self.output_size = output_size
        # 定义第一个全连接层：计算权重分数的
        # value_size1作为输出维度，是为了和Value进行三维矩阵想乘
        self.atten = nn.Linear(query_size + key_size, value_size1)
        # 定义第二个全连接层：让结果按照指定维度输出
        # 按照注意力计算步骤的第2步：如果第一步，按照拼接的方式实现的计算，那么需要将Q和第一步计算的结果再次拼接
        # 第3步：如果第二步Q和第一步计算的结果已经拼接，那么需要进行再次线性变化，得到最终的输出结果（指定输出维度）
        self.attn_combin = nn.Linear(query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        # Q--》[1,1,32]; K--》[1,1,32];V--》[1, 32, 64]
        # 实现注意力计算步骤的第一步：
        # 1.1 计算注意力权重分数:atten_weight-->[1,1,32]
        atten_weight = F.softmax(self.atten(torch.cat((Q, K), dim=-1)), dim=-1)
        # 1.2 atten_weight和V想乘得到计算结果:atten1-->[1,1,64]
        atten1 = torch.bmm(atten_weight, V)
        # 实现注意力计算步骤的第二步：
        # 2.1 需要将Q和atten1进行拼接:temp1-->[1,1,96]
        temp1 = torch.cat((Q, atten1), dim=-1)
        # 实现注意力计算步骤的第三步：对第二步的结果进行线性变化，按照指定维度输出
        result = self.attn_combin(temp1)
        return result, atten_weight



if __name__ == '__main__':
    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 32
    # my_atten = MyAtten(query_size, key_size, value_size1, value_size2, output_size)
    my_atten = GzMyAtten(query_size, key_size, value_size1, value_size2, output_size)
    Q = torch.randn(1, 1, 32)
    K = torch.randn(1, 1, 32)
    V = torch.randn(1, 32, 64)
    result, atten_weight = my_atten(Q, K, V)
    print(f'result-->{result.shape}')
    print(f'atten_weight-->{atten_weight.shape}')