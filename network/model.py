import glob
import os
import torch
from network.gsunet import GSUnet
from database.vaihingen import load_dataloader

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
    dataloader_train = load_dataloader(2, 'train')
    n_channels = 5  # NIR - R - G - DSM - nDSM
    n_classes = 6  #'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'

    model = GSUnet(n_channels,n_classes)
    data, _ = iter(dataloader_train).__next__()
    # print('data: ', data.size())
    # data  = data.to(device='cuda')

    pred = model(data)

    """
    assert pred.size(1) == len(dataloader_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataloader_train.LABEL_CLASSES)}, got {pred.size(1)})'
    assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
    assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'
    """
    return pred