# import torch
# import torch.nn as nn
# from pytorch_memlab import LineProfiler, profile, profile_every

# def test_line_report_method(device: int):
#     class Net(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = torch.nn.Linear(100, 100).cuda(device)
#             self.drop = torch.nn.Dropout(0.1)

#         @profile_every(1)
#         def forward(self, inp):
#             return self.drop(self.linear(inp))

#     net = Net()
#     inp = torch.Tensor(50, 100).cuda(device)
#     net(inp)

# if __name__ == "__main__":
#     test_line_report_method(2)
