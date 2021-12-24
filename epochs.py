from torch.nn.modules import loss
from tqdm.notebook import trange      # pretty progress bar
import matplotlib.pyplot as plt
from network.loss import JointEdgeSegLoss
import torch
import numpy as np
import cv2
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def train_epoch(data_loader, model, optimiser, device):

    # set model to training mode. This is important because some layers behave differently during training and testing
    model.train(True)
    model.to(device)

    # stats
    loss_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    with tqdm(range(len(data_loader))) as pBar:
        for idx, (data, target) in enumerate(data_loader):

            # put data and target onto correct device
            data, segmask = data.to(device), target.to(device)

            # Canny on image to get the edgemask
            x_size = segmask.size()
            segmask = np.reshape(segmask.cpu().numpy(),
                                 (x_size[0], 1, x_size[1], x_size[2]))
            im_arr = segmask.astype(np.float64)
            im_arr /= 6
            im_arr *= 255.0
            im_arr = im_arr.astype(np.uint8)
            im_arr = im_arr.transpose((0, 2, 3, 1))
            canny = np.zeros((x_size[0], x_size[1], x_size[2]))
            for i in range(x_size[0]):
                canny[i] = cv2.Canny(im_arr[i], 10, 230)
            canny = np.reshape(canny, (x_size[0], 1, x_size[1], x_size[2]))

            """plt.subplot(121),plt.imshow(segmask[1][0])
            plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(canny[1][0])
            plt.xticks([]), plt.yticks([])
            plt.show()"""

            edgemask = torch.from_numpy(canny).to(
                device).float()
            segmask = torch.from_numpy(segmask).to(
                device).float()

            # reset gradients
            optimiser.zero_grad()

            # forward pass
            segin, edgein = model(data)
            segin, edgein = segin.to(device), edgein.to(device)
            # print('segin', segin.size(), 'edgein', edgein.size())

            # loss
            n_classes = 6  # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
            criterion_joint_edgeseg_loss = JointEdgeSegLoss(
                classes=n_classes,
                mode='train').to(device)

            # return dic losses losses['seg_loss'], losses['edge_loss'], losses['att_loss'], losses['dual_loss']
            loss_dict = criterion_joint_edgeseg_loss(
                (segin, edgein), (segmask, edgemask))

            # backward pass
            loss_combined = loss_dict['seg_loss']
            loss_combined += loss_dict['edge_loss']
            loss_combined += loss_dict['att_loss']
            loss_combined += loss_dict['dual_loss']
            loss_combined.backward()

            # parameter update
            optimiser.step()

            # stats update
            loss_total += loss_combined.item()
            oa_total += torch.mean((segin.argmax(1) == segmask).to(
                device).float()).item()

            # format progress bar
            pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
                loss_total/(idx+1),
                100 * oa_total/(idx+1)
            ))
            pBar.update(1)

        pBar.close()

    # normalise stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return model, loss_total, oa_total


def validate_epoch(data_loader, model, device):       # note: no optimiser needed

  # set model to training mode. This is important because some layers behave differently during training and testing
    model.train(False)
    model.to(device)

    # stats
    loss_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    with tqdm(range(len(data_loader))) as pBar:
        for idx, (data, target) in enumerate(data_loader):

            # put data and target onto correct device
            data, segmask = data.to(device), target.to(device)

            # Canny on image to get the edgemask
            x_size = segmask.size()
            segmask = np.reshape(segmask.cpu().numpy(),
                                 (x_size[0], 1, x_size[1], x_size[2]))
            im_arr = segmask.astype(np.float64)
            im_arr /= 6
            im_arr *= 255.0
            im_arr = im_arr.astype(np.uint8)
            im_arr = im_arr.transpose((0, 2, 3, 1))
            canny = np.zeros((x_size[0], x_size[1], x_size[2]))
            for i in range(x_size[0]):
                canny[i] = cv2.Canny(im_arr[i], 10, 230)
            canny = np.reshape(canny, (x_size[0], 1, x_size[1], x_size[2]))

            edgemask = torch.from_numpy(canny).to(
                device).float()
            segmask = torch.from_numpy(segmask).to(
                device).float()

            # forward pass
            segin, edgein = model(data)
            segin, edgein = segin.to(device), edgein.to(device)

            # loss
            n_classes = 6  # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
            criterion_joint_edgeseg_loss = JointEdgeSegLoss(
                classes=n_classes,
                mode='val').to(device)
            # return dic losses losses['seg_loss'], losses['edge_loss'], losses['att_loss'], losses['dual_loss']
            loss_dict = criterion_joint_edgeseg_loss(
                (segin, edgein), (segmask, edgemask))

            # backward pass
            loss_combined = loss_dict['seg_loss']
            loss_combined += loss_dict['edge_loss']
            loss_combined += loss_dict['att_loss']
            loss_combined += loss_dict['dual_loss']
            loss_combined.backward()

            # stats update
            loss_total += loss_combined.item()
            oa_total += torch.mean((segin.argmax(1) == segmask).to(
                device).float()).item()

            # format progress bar
            pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
                loss_total/(idx+1),
                100 * oa_total/(idx+1)
            ))
            pBar.update(1)

        pBar.close()

    # normalise stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return model, loss_total, oa_total
