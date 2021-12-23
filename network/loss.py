import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


"""
criterion = nn.CrossEntropyLoss()

criterion_img_wt_loss = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, size_average=True,
            ignore_index=args.dataset_cls.ignore_label, 
            upper_bound=args.wt_bound).cuda()

criterion_crossentropy_loss = CrossEntropyLoss2d(size_average=True, ignore_index=args.dataset_cls.ignore_label).cuda()

criterion_joint_edgeseg_loss = JointEdgeSegLoss(classes=args.dataset_cls.num_classes,
           ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
           edge_weight=args.edge_weight, seg_weight=args.seg_weight, att_weight=args.att_weight, dual_weight=args.dual_weight).cuda()

criterion_val = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, mode='val',
       ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
       edge_weight=args.edge_weight, seg_weight=args.seg_weight).cuda()
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train',
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        elif mode == 'val':
          # reduction='mean' instead of size_average
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss()

    def bce2d(self, input, target):
        # input.size() == nb_batch x 1 x w x h
        # target.size() == nb_batch x 1 x w x h
        """print('JointEdgeSegLoss bce2d')
        print('input', input.size())
        print('target', target.size())"""

        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(
            2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        # reduction='mean' instead of size_average
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        # input.size() == nb_batch x 6 x w x h
        # target.size() == nb_batch x 1 x w x h
        # edge.size() == nb_batch x 1 x w x h
        """print('JointEdgeSegLoss edge_attention')
        print('input', input.size())
        print('target', target.size())
        print('edge', edge.size())"""

        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        """print('where    ',torch.where(edge > 0.8, target, filler))
        plt.subplot(131),plt.imshow(input[1][0].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(torch.where(edge > 0.8, target, filler)[1][0].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(edge.max(1)[0][1].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.show()"""

        # return self.seg_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))
        return self.seg_loss(input, torch.where(edge > 0.8, target, filler))

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        """print('!!!!!!!!')
        print('segin', segin.size())
        print('edgein', edgein.size())
        print('segmask', segmask.size())
        print('edgemask', edgemask.size())"""

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * \
            20 * self.bce2d(edgein, edgemask)
        losses['att_loss'] = self.att_weight * \
            self.edge_attention(segin, segmask, edgein)
        losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask, ignore_pixel=5)

        return losses


class ImageBasedCrossEntropyLoss2d(nn.Module):

    # reduction='mean' instead of size_average
    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(
            weight, size_average, ignore_index)  # NLLLoss instead
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = True

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        # input.size() == nb_batch x 6 x w x h
        # target.size() == nb_batch x 1 x w x h
        
        """print('ImageBasedCrossEntropyLoss2d forward')
        print('inputs', inputs.size())
        print('targets', targets.size())"""


        target_cpu = targets.data.cpu().numpy()


        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in [1, 1]:# range(0, inputs.shape[0]):    # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            # loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)), targets[i].unsqueeze(0))
            for j in range(0, inputs[i].shape[0]):
                temp = target_cpu[i][0] == j

                """plt.subplot(121),plt.imshow(F.log_softmax(inputs[i][j].unsqueeze(0)).data.cpu().numpy())
                plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(temp)
                plt.xticks([]), plt.yticks([])
                plt.show()"""

                # print(F.log_softmax(inputs[i][j]).data.size())
                # print(torch.Tensor(temp).to(device).unsqueeze(0).size())
                """
                # EXAMPLE
                # 2 batches, 3 input channels, 1 x 5
                loss_nll = nn.NLLLoss2d()
                target = torch.Tensor(2, 1, 5).random_(3).long()
                data = torch.rand(2, 3, 1, 5)
                input1 = torch.Tensor(data, requires_grad=True)
                loss1 = loss_nll(F.log_softmax(input1), target)
                """
        # print(np.squeeze(targets, axis=1).shape)
        m = nn.LogSoftmax(dim=1)
        loss += self.nll_loss(m(inputs), np.squeeze(targets, axis=1).long())
        # print('loss', loss)
        return loss


class CrossEntropyLoss2d(nn.Module):
  # reduction='mean' instead of size_average
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(
            weight, size_average, ignore_index)  # NLLLoss isntead

    def forward(self, inputs, targets):
        # input.size() == nb_batch x 6 x w x h
        # target.size() == nb_batch x 1 x w x h
        """print('CrossEntropyLoss2d forward')
        print('inputs', inputs.size())
        print('targets', targets.size())"""
        return self.nll_loss(F.log_softmax(inputs), targets)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    """print(np.squeeze(labels, axis=1).shape)
    
    print(y[np.squeeze(labels, axis=1)].permute(0, 3, 1, 2).shape)"""
    return y[np.squeeze(labels, axis=1)].permute(0, 3, 1, 2)


def gradient_central_diff(input, cuda):
    return input, input


def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y


def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input.shape
    return input


def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox, Ox) + torch.mul(Oy, Oy) + 1e-6)
    mag = mag / mag.max()

    return mag


class DualTaskLoss(nn.Module):
    def __init__(self, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        return

    def forward(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        # input_logits.size() == nb_batch x 6 x w x h
        # gts.size() == nb_batch x 1 x w x h
        """print('DualTaskLoss forward')
        print('input_logits', input_logits.size())
        print('gts', gts.size())"""

        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        ignore_mask = (gts == ignore_pixel).detach()

        # print('!!!!ignore_mask', ignore_mask.shape)

        input_logits = torch.where(ignore_mask.view(N, 1, H, W).expand(N, C, H, W),
                                   torch.zeros(N, C, H, W).cuda(),
                                   input_logits)

        # print('!!!!input_logits', input_logits.shape)

        """plt.subplot(131),plt.imshow(ignore_mask[1][0].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(input_logits[1][0].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.show()"""

        gt_semantic_masks = gts.detach().long()
        # print('!!!!gt_semantic_masks', gt_semantic_masks.shape)
        gt_semantic_masks = torch.where(ignore_mask, torch.zeros(
            N, 1, H, W).long().cuda(), gt_semantic_masks).long()

        # print('!!!!gt_semantic_masks', gt_semantic_masks.shape)
        gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, C).detach()
        """
        plt.subplot(131),plt.imshow(gt_semantic_masks[1][4].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(gt_semantic_masks[1][5].data.cpu().numpy())
        plt.xticks([]), plt.yticks([])
        plt.show()"""
        # print('!!!!gt_semantic_masks', gt_semantic_masks.shape)

        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)
        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)

        """print('!!!!g', g.shape)
        print('!!!!g_hat', g_hat.shape)"""

        """g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)"""
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(
            loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(
            loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat
        # print('total_loss', total_loss)

        return total_loss
