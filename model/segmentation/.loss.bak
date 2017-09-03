import torch
import torch.nn as nn
import torch.nn.functional as F
# turn into pytorch

#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

#https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

        #
        # logits_flat  = logits.view (-1)
        # targets_flat = targets.view(-1)
        # return StableBCELoss()(logits_flat,targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score

class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss



class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score