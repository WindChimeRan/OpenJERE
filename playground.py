import torch
import torch.nn as nn

if __name__ == "__main__":
    rnn = nn.GRU(input_size=300, hidden_size=300, batch_first=True, bidirectional=False)
    input = torch.randn(10000, 1, 300) # b 5, s 3, h 10
    h0 = torch.randn(1, 10000, 300) # s 1, n 5
    output, hn = rnn(input, h0)
    # print(hn.size())
    # print(output.size())
    print(input.permute(1,0,2).size())