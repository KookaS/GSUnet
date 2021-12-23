import torch
import torch.nn.functional as F
from torch import nn

import cv2
import numpy as np
import matplotlib.pyplot as plt

from network.other import BasicBlock, DoubleConv, GatedSpatialConv2d, _AtrousSpatialPyramidPoolingModule

from pylab import arange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GSUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GSUnet, self).__init__()

        # down
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        # shape stream
        # self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1, device=device)
        self.dsn3 = nn.Conv2d(256, 1, 1, device=device)
        self.dsn4 = nn.Conv2d(512, 1, 1, device=device)
        self.dsn5 = nn.Conv2d(1024, 1, 1, device=device)

        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1, device=device)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1, device=device)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1, device=device)
        self.res4 = BasicBlock(8, 8, stride=1, downsample=None)
        self.d4 = nn.Conv2d(8, 4, 1, device=device)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False, device=device)

        self.gate1 = GatedSpatialConv2d(32, 32).to(device)
        self.gate2 = GatedSpatialConv2d(16, 16).to(device)
        self.gate3 = GatedSpatialConv2d(8, 8).to(device)
        self.gate4 = GatedSpatialConv2d(4, 4).to(device)
        # self.gate5 = GatedSpatialConv2d(2, 2)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

        # fusion
        reduction = 128
        # _AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)
        self.aspp = _AtrousSpatialPyramidPoolingModule(
            1, reduction, output_stride=8).to(device)

        # up
        self.up6 = nn.ConvTranspose2d(reduction*6, 512, 2, stride=2).to(device)
        self.conv6 = DoubleConv(1024, 512).to(device)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2).to(device)
        self.conv7 = DoubleConv(512, 256).to(device)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2).to(device)
        self.conv8 = DoubleConv(256, 128).to(device)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2).to(device)
        self.conv9 = DoubleConv(128, 64).to(device)
        self.conv10 = nn.Conv2d(64, out_ch, 1).to(device)

    def forward(self, inputs, gts=None):
        x_size = inputs.size()
        inputs = inputs.to(device)

        # down
        c1 = self.conv1(inputs)
        #print('c1', c1.size())
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        #print('c2', c2.size())
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        #print('c3', c3.size())
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        #print('c4', c4.size())
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        #print('c5', c5.size())

        # shape stream

        # conv 1x1
        """
        s1 = F.interpolate(c1, x_size[2:], mode='bilinear', align_corners=True)
        print('s1', s1.size())
        s2 = F.interpolate(self.dsn2(c2), x_size[2:], mode='bilinear', align_corners=True)
        print('s2', s2.size())
        s3 = F.interpolate(self.dsn3(c3), x_size[2:], mode='bilinear', align_corners=True)
        print('s3', s3.size())
        s4 = F.interpolate(self.dsn4(c4), x_size[2:], mode='bilinear', align_corners=True)
        print('s4', s4.size())
        # s5 = F.interpolate(self.dsn5(c5), x_size[2:], mode='bilinear', align_corners=True)
        """
        s1 = F.interpolate(c1, c1.size()[2:],
                           mode='bilinear', align_corners=True)
        #print('s1', s1.size())
        s2 = F.interpolate(self.dsn2(c2), c2.size()[
                           2:], mode='bilinear', align_corners=True)
        #print('s2', s2.size())
        s3 = F.interpolate(self.dsn3(c3), c3.size()[
                           2:], mode='bilinear', align_corners=True)
        #print('s3', s3.size())
        s4 = F.interpolate(self.dsn4(c4), c4.size()[
                           2:], mode='bilinear', align_corners=True)
        #print('s4', s4.size())
        s5 = F.interpolate(self.dsn5(c5), c5.size()[
                           2:], mode='bilinear', align_corners=True)
        #print('s5', s5.size())

        ## res & gated
        """
        cs = self.res1(s1)
        print('res1', cs.size())
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        print('interp1', cs.size())
        cs = self.d1(cs)
        print('d1', cs.size())
        cs = self.gate1(cs, s2)
        print('gate1', cs.size())

        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s3)

        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s4)
        """
        cs = self.res1(s1)
        #print('res1', cs.size())
        cs = F.interpolate(cs, s2.size()[2:],
                           mode='bilinear', align_corners=True)
        #print('interp1', cs.size())
        cs = self.d1(cs)
        #print('d1', cs.size())
        cs = self.gate1(cs, s2)
        #print('gate1', cs.size())

        cs = self.res2(cs)
        cs = F.interpolate(cs, s3.size()[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s3)

        cs = self.res3(cs)
        cs = F.interpolate(cs, s4.size()[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s4)

        cs = self.res4(cs)
        cs = F.interpolate(cs, s5.size()[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d4(cs)
        cs = self.gate4(cs, s5)

        # end of shape stream
        """im_arr = inputs.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).to(device).float()
        #print('canny', canny.size())"""

        im_arr = inputs[:, 4, :, :].cpu().numpy().astype(np.float64) # computes Canny on the 5th image
        im_arr = np.reshape(im_arr, (x_size[0], 1, x_size[2], x_size[3])).astype(np.float64)
        original = im_arr.copy()
        for i in range(x_size[0]):
            # im_arr[i] /= im_arr[i].max()
            # im_arr[i] += np.abs(im_arr[i].min())
            # im_arr[i] *= 1000
            """
            maxIntensity = im_arr[i].max()
            phi = 10.0
            theta = 1.0
            im_arr[i] = (maxIntensity/phi)*(im_arr[i]/(maxIntensity/theta))**2
            """
            im_arr[i] *= 200
            """
            kernel = np.ones((20,20),np.float64)/(20*20)
            im_arr[i] = cv2.filter2D(im_arr[i],-1,kernel)
            """
            im_arr[i] = cv2.bilateralFilter(im_arr[i][0].astype(np.uint8), 15, 200, 100)   # source, diameter, sigma_color, sigma_space
        temp = im_arr
        im_arr = im_arr.astype(np.uint8)
        im_arr = im_arr.transpose((0, 2, 3, 1))
        canny = np.zeros((x_size[0], x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], im_arr[i].min(), im_arr[i].max())  
        canny = np.reshape(canny, (x_size[0], 1, x_size[2], x_size[3]))
        canny = torch.from_numpy(canny).to(device).float()


        plt.subplot(141),plt.imshow(original[1][0]) #inputs.cpu().numpy().astype(np.float64)[1, 4, :, :])
        plt.xticks([]), plt.yticks([])
        plt.subplot(142),plt.imshow(temp[1][0])
        plt.xticks([]), plt.yticks([])
        plt.subplot(143),plt.imshow(canny.cpu().numpy().astype(np.float64)[1][0])
        plt.xticks([]), plt.yticks([])
        plt.subplot(144),plt.imshow(inputs.cpu().numpy().astype(np.float64)[1, 0, :, :])
        plt.xticks([]), plt.yticks([])
        plt.show()


        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        # cs = F.interpolate(cs, s5.size()[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        #print('acts', acts.size())

        # fusion
        astr = self.aspp(s5, acts)
        #print('astr', astr.size())

        # up
        up_6 = self.up6(astr)
        #print('up_6', up_6.size())
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        seg_out = self.conv10(c9)

        return seg_out, edge_out
        """
        if self.training:
            return self.criterion((seg_out, edge_out), gts)              
        else:
            return seg_out, edge_out
        """

"""
"""