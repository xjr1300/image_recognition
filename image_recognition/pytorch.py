import torch
import torch.nn.functional as F
from torch import nn

t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.zeros((32, 3, 128, 128))

print(f"t1 = {t1}, t1.shape = {t1.shape}")
print(f"t2.shape = {t2.shape}")

# t1 = torch.tensor([1, 2, 3, 4], device="cuda")
# t2 = torch.tensor([1, 2, 3, 4])
# t2 = t2.to("cuda")

t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([2, 4, 6, 8])
t3 = t1 + t2
t4 = t1**2
print(f"t3 = {t3}")
print(f"t4 = {t4}")

t1 = torch.tensor([1, 2, 3, 4])
t1 = t1.view(2, 2)
print(f"t1 = {t1}")
t2 = torch.tensor([1, 2, 3, 4, 5, 6])
t2 = t2.view(2, -1)
print(f"t2 = {t2}")

t1 = torch.zeros((32, 3, 128, 128))
t1 = t1.transpose(0, 2)
print(f"t1.shape = {t1.shape}")
t2 = torch.zeros((32, 3, 128, 128))
t2 = t2.permute(2, 0, 3, 1)
print(f"t2.shape = {t2.shape}")

t1 = torch.tensor([1, 2, 3, 4, 5, 6]).view(2, 3)
t2 = torch.tensor([7, 8, 9]).view(1, 3)
t3 = torch.cat((t1, t2))
print(f"t3 = {t3}")
t4 = torch.tensor([1, 2, 3])
t5 = torch.tensor([4, 5, 6])
t6 = torch.stack((t4, t5), dim=1)
print(f"t6 = {t6}")

t1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(3, 3)
print(f"t1 = {t1}")
t2 = t1[[0, 1]]
print(f"t2 = {t2}")
t3 = t1[:, [0, 2]]
print(f"t3 = {t3}")
t4 = t1[[0, 2, 1], [1, 2, 1]]
print(f"t4 = {t4}")
t5 = t1[[True, False, False]]
print(f"t5 = {t5}")
t6 = t1[t1 % 2 == 0]
print(f"t6 = {t6}")
