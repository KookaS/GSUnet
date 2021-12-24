import torch
from database.vaihingen import visualise, load_dataloader
from network.optimizer import setup_optimiser
from network.model import test_model, load_model, save_model, evaluate_model
from epochs import train_epoch, validate_epoch
from network.device import get_device

def main():

    seed = 323444           # the seed value used to initialise the random number generator of PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # visualise()
    # pred = test_model()

    # define hyperparameters
    device = get_device()
    print('device used for computation: ', device)
    start_epoch = 0        # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
    batch_size = 2
    learning_rate = 0.1
    momentum = 0.5
    weight_decay = 0.001
    num_epochs = 40
    n_channels = 5  # NIR - R - G - DSM - nDSM
    n_classes = 6  #'Impervious', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter'

    # initialise data loaders
    dl_train = load_dataloader(batch_size, 'train')
    dl_val = load_dataloader(batch_size, 'val')

    # load model
    model, epoch = load_model(n_channels, n_classes,epoch=start_epoch)
    optimi = setup_optimiser(model, learning_rate, momentum, weight_decay)
    
    # multiply learning rate by 0.1 after 30% of epochs
    # scheduler = optim.lr_scheduler.StepLR(optimi, step_size=int(0.3*num_epochs), gamma=0.1)

    # do epochs
    while epoch < num_epochs:

        # training
        model, loss_train, oa_train = train_epoch(dl_train, model, optimi, device)

        # validation
        loss_val, oa_val = validate_epoch(dl_val, model, device)

        # print stats
        print('[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}'.format(
            epoch+1, num_epochs,
            loss_train, loss_val,
            100*oa_train, 100*oa_val
        ))

        # save model
        epoch += 1
        save_model(model, epoch)
    
    """
    # visualize predictions for a number of epochs
    dl_val_single = load_dataloader(1, 'val')
    epochs = [0, 15, 20, 36]
    n_channels = 5
    n_classes = 6
    evaluate_model(dl_val_single, n_channels, n_classes,epochs, dataset_train=dl_train, show_metrics = True, numImages=15,device = device)
    """
    


if __name__ == '__main__':
    main()