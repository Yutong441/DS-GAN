from torch import nn
import torch.nn.functional as F


class ECA(nn.Module):
    '''
    modified from https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/
    '''

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.fc = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=k_size,
                          padding=(k_size - 1) // 2, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        N = len(x.size()) - 2
        y = F.adaptive_avg_pool2d(x, 1).view(x.size()[:2])
        # Two different branches of ECA module
        y = self.fc(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        new_size = list(x.size()[:2]) + [1]*N
        y = y.view(*new_size)
        return x * y.expand_as(x)
