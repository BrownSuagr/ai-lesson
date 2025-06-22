import torch
a = torch.randn(2, 3, 4)
print(a)
print(a[:2].shape) # [2,3,4]
print(a[:2, :].shape) #[2, 3, 4]
print(a[:2, :, :].shape) #[2, 3, 4]

# print(a[2, :, :].shape)# 报错
print(a[1, :, :].shape)# [3, 4]
print(a[1, :, :])# [3, 4]

print(a[0, :2].shape) # [2, 4]
print(a[0, :2, 1].shape) # [2]
print(a[0, :2, 1]) # [2]

# 张量的切片形状：有几个冒号就是几维，但是如果假如你是3维度，你没有写全3维，剩下的维度，默认全取（冒号）