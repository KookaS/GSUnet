import glob
import os
import torch
from network.gsunet import GSUnet
from database.vaihingen import load_dataloader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
from network.metrics import compute_metrics
import platform
from database.vaihingen import get_labels

os.makedirs('cnn_states/GSUnet', exist_ok=True)


def load_model(n_channels=5, n_classes=6, epoch='latest'):
    model = GSUnet(n_channels, n_classes)
    modelStates = glob.glob('cnn_states/GSUnet/*.pth')
    if len(modelStates) and (epoch == 'latest' or epoch > 0):

        if platform.system() == 'Windows':
            modelStates = [m.replace('cnn_states/GSUnet', '')[1:].replace('.pth', '') for m in modelStates]
        else:
            modelStates = [int(m.replace('cnn_states/GSUnet/', '').replace('.pth', '')) for m in modelStates]
        if epoch == 'latest':
            epoch = max(modelStates)
        stateDict = torch.load(
            open(f'cnn_states/GSUnet/{epoch}.pth', 'rb'), map_location='cpu')
        model.load_state_dict(stateDict)
    else:
        # fresh model
        epoch = 0
    return model, epoch


def save_model(model, epoch):
    torch.save(model.state_dict(), open(
        f'cnn_states/GSUnet/{epoch}.pth', 'wb'))


def test_model():
    dataloader_train = load_dataloader(batch_size=2, split='train')
    n_channels = 5  # NIR - R - G - DSM - nDSM
    n_classes = 6  # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'

    model = GSUnet(n_channels, n_classes)
    data, _ = iter(dataloader_train).__next__()

    pred = model(data)

    """
    assert pred.size(1) == len(dataloader_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataloader_train.LABEL_CLASSES)}, got {pred.size(1)})'
    assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
    assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'
    """
    return pred

def evaluate_model(dataLoader, n_channels, n_classes, epochs, show_metrics=False, numImages=5, device=None):
    models = [load_model(n_channels, n_classes, e)[0] for e in epochs]
    numModels = len(models)
    cMap = ListedColormap(['black', 'grey', 'lawngreen', 'darkgreen', 'orange',
                            'red'])  # 'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
    for idx, (data, labels) in enumerate(dataLoader):
        list_gt_labels = []
        if idx == 0:
            continue
        if idx == numImages:
            break

        f, ax = plt.subplots(nrows=1, ncols=numModels + 1, figsize=(15, 15))

        list_gt_labels.append(labels[0, ...].cpu().numpy().flatten())

        # plot ground truth
        ax[0].imshow(labels[0, ...].cpu().numpy(), cmap=cMap)
        ax[0].axis('off')
        ax[0].set_title('Ground Truth')
        conf_matrix = []
        accuracy = []
        for mIdx, model in enumerate(models):
            list_predictions = []
            model = model.to(device)

            with torch.no_grad():
                seg_out, edge_out = model(data.to(device))
                yhat = seg_out.data.max(1)[1]


                list_predictions.append(yhat[0, ...].flatten())
                all_predictions = np.concatenate(list_predictions)
                all_gt_labels = np.concatenate(list_gt_labels)
                accuracy.append(accuracy_score(all_gt_labels, all_predictions))
                conf_matrix.append(confusion_matrix(all_gt_labels, all_predictions))

                # plot model predictions
                ax[mIdx + 1].imshow(yhat[0, ...].cpu().numpy(), cmap=cMap)
                ax[mIdx + 1].axis('off')
                cax = ax[mIdx + 1].set_title(f'Epoch {epochs[mIdx]}')

        if show_metrics:
            _, ax = plt.subplots(nrows=1, ncols=numModels, figsize=(20, 20))
            for mIdx, model in enumerate(models):
                conf_matrix_one = conf_matrix[mIdx]

                ax[mIdx].matshow(conf_matrix_one, cmap=plt.cm.Blues, alpha=0.5)

                iou, recall, precision, f1, kappa = compute_metrics(conf_matrix_one)

                for i in range(conf_matrix_one.shape[0]):
                    for j in range(conf_matrix_one.shape[1]):
                        if math.isnan(conf_matrix_one[i, j]):
                            conf_matrix_one[i, j] = 0
                        ax[mIdx].text(x=j, y=i, s=conf_matrix_one[i, j], va='center', ha='center', size='x-large')
                    ax[mIdx].set_xlabel('Predictions', fontsize=18)
                    ax[mIdx].set_ylabel('Ground Truth', fontsize=18)
                    ax[mIdx].set_title('Confusion Matrix', fontsize=18)
                if idx == 1 and epochs[mIdx] == 'latest':
                    print("F1", f1)
                    print("IoU", iou)
                    print("Kappa", kappa)
                    # print("OA", OA)
                    # print("UA", UA)
                    # print("PA", PA)
                    print("Mean FScore", sum(f1) / len(f1))
                    print("Mean IoU", sum(iou) / len(iou))
                    # print("Mean UA", sum(UA) / len(UA))
                    # print("Mean PA", sum(PA) / len(PA))
        plt.show()
