import glob
import os
import torch
from network.gsunet import GSUnet
from database.vaihingen import load_dataloader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.makedirs('cnn_states/GSUnet', exist_ok=True)

def load_model(n_channels=5, n_classes=6, epoch='latest'):
    model = GSUnet(n_channels, n_classes)
    modelStates = glob.glob('cnn_states/GSUnet/*.pth')
    if len(modelStates) and (epoch == 'latest' or epoch > 0):
        modelStates = [int(m.replace(
            'cnn_states/GSUnet/', '').replace('.pth', '')) for m in modelStates]
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
    n_classes = 6  #'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'

    model = GSUnet(n_channels,n_classes)
    data, _ = iter(dataloader_train).__next__()

    pred = model(data)

    """
    assert pred.size(1) == len(dataloader_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataloader_train.LABEL_CLASSES)}, got {pred.size(1)})'
    assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
    assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'
    """
    return pred

def evaluate_model(dataLoader, n_channels, n_classes, epochs, dataset_train, numImages=5, device=None):
    models = [load_model(n_channels, n_classes, e)[0] for e in epochs]
    numModels = len(models)
    for idx, (data, labels) in enumerate(dataLoader):
        if idx == numImages:
            break

    _, ax = plt.subplots(nrows=1, ncols=numModels+1, figsize = (20, 15))

    # plot ground truth
    ax[0].imshow(labels[0,...].cpu().numpy())
    ax[0].axis('off')
    if idx == 0:
        ax[0].set_title('Ground Truth')

    for mIdx, model in enumerate(models):
        model = model.to(device)
        with torch.no_grad():
            pred = model(data.to(device))

            # get the label (i.e., the maximum position for each pixel along the class dimension)
            #yhat = torch.nn.functional.softmax(pred, dim = 1)
            yhat = torch.argmax(pred, dim=1)
            #y = torch.squeeze(yhat.cpu()).numpy() # 256 256
                
            #plt.imshow(y)
            cMap = ListedColormap(['black', 'grey', 'lawngreen', 'darkgreen', 'orange', 'red'])     #  'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'
            
            # plot model predictions
            ax[mIdx+1].imshow(yhat[0,...].cpu().numpy(), cmap=cMap)
            ax[mIdx+1].axis('off')
            if idx == 0:
                cax = ax[mIdx+1].set_title(f'Epoch {epochs[mIdx]}')
                cbar = f.colorbar(cax, ticks=list(range(len(dataset_train.LABEL_CLASSES))))
                cbar.ax.set_yticklabels(list(dataset_train.LABEL_CLASSES))
                

