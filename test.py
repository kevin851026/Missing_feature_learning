import torch
from torch import nn
import torch.nn.functional as F
class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss
c = softCrossEntropy()
a = torch.Tensor([[1.,2.,3.]])
b = torch.Tensor([[0.001,0.149,0.85]])

# print(b/10)
print(F.softmax(b/3))
# print(c(a,b))