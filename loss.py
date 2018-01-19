import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SegmentationLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, gen_segmentation, gt_segmentation):
        segm_loss = self.nll_loss(F.log_softmax(gen_segmentation, dim=1), gt_segmentation)
        return segm_loss


class ImageLoss(nn.Module):
    def __init__(self):
        super(ImageLoss, self).__init__()

    def forward(self, gen_image, gt_image):
        l2_loss = (gen_image - gt_image) ** 2
        return l2_loss.mean()
