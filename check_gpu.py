import torch
a = torch.randn(10).cuda()
print(a)
print(a + 2)
print(torch.cuda.is_available())