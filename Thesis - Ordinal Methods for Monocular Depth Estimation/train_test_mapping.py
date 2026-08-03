
################ LIBRARIES AND ARGUMENTS ################

import argparse
def str_to_bool(value):
    if value.lower() in ('yes', 'true', '1'):
        return True
    elif value.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid value for boolean argument. Accepted values are "yes"/"true"/"1" or "no"/"false"/"0".')

parser = argparse.ArgumentParser()
parser.add_argument('SEED', type=int)
parser.add_argument('DATASET')
parser.add_argument('MODEL')
parser.add_argument('LOSS')
parser.add_argument('DISCRETIZER')
parser.add_argument('NCLASSES', type=int)
parser.add_argument('BATCH', type=int)
parser.add_argument('EPOCHS', type=int)
parser.add_argument('TRAIN_MODEL', type=str_to_bool, help='Specify whether to train the model (yes/true/1) or not (no/false/0)')
args = parser.parse_args()


import dataset
import deep_ordinal as ordinal_losses
import metrics
import networks
import utils
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW
import time
import matplotlib.pyplot as plt
import torchmetrics
import csv
import os

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


################ VARIABLES AND ERRORS CONTROL ################

N_CLASSES = args.NCLASSES
SELECTED_DATASET = args.DATASET
SELECTED_DISCRETIZER = args.DISCRETIZER
SELECTED_NETWORK = args.MODEL
LOSS_NAME = args.LOSS
TRAIN_MODEL = args.TRAIN_MODEL
BATCH_SIZE = args.BATCH
SEED = args.SEED

# to ensure reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)

if SELECTED_DATASET == "Kitti" or SELECTED_DATASET == "KittiMono":
    IMAGE_WIDTH = 1238
    IMAGE_HEIGHT = 374
    MAX_DEPTH = 80 # (in meters) Velodyne HDL64E (120m max range)
    MIN_DEPTH = 1
else:
    assert False, "Error, check dataset classes."
    
if SELECTED_NETWORK == "DeepLab" or SELECTED_NETWORK == "FCN" or SELECTED_NETWORK == "LRASPP" or SELECTED_NETWORK == "DORN" or SELECTED_NETWORK == "RSIDE":
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
else:
    assert False, "Error, check network classes."

if SELECTED_DISCRETIZER == "SID":
    BOUNDARIES = np.array(utils.SID(N_CLASSES-1, MIN_DEPTH, MAX_DEPTH))
    print("Boundaries: ", BOUNDARIES)
elif SELECTED_DISCRETIZER == "UD":
    BOUNDARIES = np.array(utils.UD(N_CLASSES-1, MIN_DEPTH, MAX_DEPTH))
    print("Boundaries: ", BOUNDARIES)
else:
    assert False, "Error, check discretizer definitions."
    
DEPTH_RANGES = torch.tensor(utils.classes_to_depth(MIN_DEPTH, MAX_DEPTH, BOUNDARIES)).to(dtype=torch.float, device=device)
print("Depth ranges: ", DEPTH_RANGES)


DEPTH_RANGES[None,:]
################ TRANSFORMATIONS ################

transform_aug = A.Compose([A.Resize(int(IMAGE_HEIGHT*1.1), int(IMAGE_WIDTH*1.1), p=1.0),
                            A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH),
                            A.HorizontalFlip(p=0.5),
                            A.CLAHE(clip_limit=4.0, tile_grid_size=(2,2), p=0.5),
                            A.Normalize(mean=MEAN, std=STD),
                            ToTensorV2()])

transform_base = A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH, p=1.0),
                            A.Normalize(mean=MEAN, std=STD),
                            ToTensorV2()])


if SELECTED_DATASET == "Kitti":
    NUM_TOTAL_TRAINING_SAMPLES = 7481
    num_tr_samples = int(0.95*NUM_TOTAL_TRAINING_SAMPLES)
    data = list(range(0, NUM_TOTAL_TRAINING_SAMPLES))
    tr_dataset = dataset.Kitti(r'/data/auto/kitti/', 'training', BOUNDARIES, data[:num_tr_samples], transform_aug)
    vl_dataset = dataset.Kitti(r'/data/auto/kitti/', 'training', BOUNDARIES, data[num_tr_samples:], transform_base)
    ts_dataset = dataset.Kitti(r'/data/auto/kitti/', 'testing', BOUNDARIES, dict_transform=transform_base)
elif SELECTED_DATASET == "KittiMono":
    tr_dataset = dataset.KittiMono(r'/data/auto/kitti/', 'training', BOUNDARIES, transform_aug)
    vl_dataset = dataset.KittiMono(r'/data/auto/kitti/', 'validation', BOUNDARIES, transform_base)
    ts_dataset = dataset.KittiMono(r'/data/auto/kitti/', 'testing', BOUNDARIES, dict_transform=transform_base)

else:
    assert False, "Error, check dataset classes."

print("Total number of images in tr_dataset:", len(tr_dataset), "; Total number of images in vl_dataset:", len(vl_dataset), "; Total number of images in ts_dataset:", len(ts_dataset))
tr = DataLoader(tr_dataset, BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn)
vl = DataLoader(vl_dataset, BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)
ts = DataLoader(ts_dataset, BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)

# show data
image_tr_show, range_tr_show, range_disc_tr_show, range_disc_valid_tr_show = tr_dataset.__getitem__(0)



################ NEURAL NETWORK ################

# select loss function
ordinal_method = getattr(ordinal_losses, LOSS_NAME)(K=N_CLASSES).to(device)
n_outputs = ordinal_method.how_many_outputs()
print('Selected loss function:', ordinal_method)
print('Loss output:', n_outputs)

# networks
if SELECTED_NETWORK == "DeepLab":
    net = networks.DeepLab(n_outputs, 'ResNet101')
elif SELECTED_NETWORK == "FCN":
    net = networks.FCN(n_outputs, 'ResNet101')
elif SELECTED_NETWORK == "LRASPP":
    net = networks.LRASPP(n_outputs)
elif SELECTED_NETWORK == "DORN":
    net = networks.DORN_ResNet101(n_outputs, (image_tr_show.shape[1], image_tr_show.shape[2]))
elif SELECTED_NETWORK == "RSIDE":
    net = networks.model(networks.E_resnet(networks.resnet50(pretrained=True)), num_features=2048, block_channel=[256, 512, 1024, 2048], K=n_outputs)
else:
    assert False, "Error, check network classes."

net.to(device)

# mapping
mapping_net = networks.OrdinalMapping(n_outputs, ordinal_method)

# select optimizer and lr
depth_optimizer = AdamW(mapping_net.parameters(), lr=1e-5, weight_decay=1e-4)

# path to save the models
path_best_model = "../models/MDE_AD_Best_{}_{}_{}_{}_{}.pth".format(net.__class__.__name__, ordinal_method.__class__.__name__, tr_dataset.__class__.__name__, SELECTED_DISCRETIZER, N_CLASSES, 'Mapping')
path_last_model = "../models/MDE_AD_Last_{}_{}_{}_{}_{}.pth".format(net.__class__.__name__, ordinal_method.__class__.__name__, tr_dataset.__class__.__name__, SELECTED_DISCRETIZER, N_CLASSES, 'Mapping')



################ TRAINING ################

if(TRAIN_MODEL):
    print("[INFO] Network training and validation...")
    EPOCHS = args.EPOCHS # number of EPOCHS
    PATIENCE = int(EPOCHS*0.8)
    VALID_LOSS_MIN = 1e6
    WAIT = 0
    loss_avg_tr = []
    loss_avg_vl = []

    # loop over EPOCHS
    for epoch in range(EPOCHS):
        print(f'* Epoch {epoch+1}/{EPOCHS}')
        
        loss_total_tr_depth = 0
        loss_total_vl_depth = 0
        
        tic = time.time()
        net.train()
        ordinal_mapping = networks.OrdinalMapping(K=n_outputs) # is it here?

        for image_tr, range_tr, range_disc_tr, range_disc_valid_tr in tr:

            image_tr = image_tr.to(device)
            range_tr = range_tr.to(device).flatten()
            range_disc_tr = range_disc_tr.to(device).flatten()
            range_disc_valid_tr = range_disc_valid_tr.to(device).flatten()

            pred_tr = net(image_tr)
            pred_tr = pred_tr.permute(0, 2, 3, 1).flatten(0, 2)

            probs_tr = ordinal_method.to_probabilities(pred_tr)

            pred_tr_depths = ordinal_mapping.forward(probs_tr)

            if LOSS_NAME == "MSE_continuous":
                range_disc_tr = range_tr # do not discritize for MSE_continuous
                
            ## forward
            # compute loss
            # loss_tr_ordinal = (range_disc_valid_tr * ordinal_method.compute_loss(pred_tr, range_disc_tr)).sum() / range_disc_valid_tr.sum()
            
            loss_tr_depth = torch.mean((range_disc_valid_tr * (pred_tr_depths - range_tr))**2)
            loss_total_tr_depth += loss_tr_depth.item()
                    
            ## backward
            # zero the gradients
            depth_optimizer.zero_grad()
            
            # compute gradients
            loss_tr_depth.backward()
            
            # adjust learning weights
            depth_optimizer.step()
            
        toc = time.time()
        print(f'  Elapsed training time: {toc-tic}s')
        

        # Validation
        tic = time.time()
        net.eval() # or net.train(False)
        ordinal_mapping = networks.OrdinalMapping(K=n_outputs)
        
        with torch.no_grad():
            for image_vl, range_vl, range_disc_vl, range_disc_valid_vl in vl:
                image_vl = image_vl.to(device)
                range_vl = range_vl.to(device).flatten()
                range_disc_vl = range_disc_vl.to(device).flatten()
                range_disc_valid_vl = range_disc_valid_vl.to(device).flatten()

                pred_vl = net(image_vl)
                pred_vl = pred_vl.permute(0, 2, 3, 1).flatten(0, 2) # 4D => 2D

                probs_vl = ordinal_method.to_probabilities(pred_vl)

                pred_vl_depths = ordinal_mapping.forward(probs_vl)

                if LOSS_NAME == "MSE_continuous":
                    range_disc_vl = range_vl # do not discritize for MSE_continuous

                # forward
                # loss_vl_ordinal = (range_disc_valid_vl * ordinal_method.compute_loss(pred_vl, range_disc_vl)).sum() / range_disc_valid_vl.sum()
                loss_vl_depth = torch.mean((range_disc_valid_vl *(pred_vl_depths - range_vl))**2)
                loss_total_vl_depth += loss_vl_depth.item()
                
        toc = time.time()
        
        loss_avg_tr.append(loss_total_tr_depth / len(tr))
        loss_avg_vl.append(loss_total_vl_depth / len(vl))

        print(f'Elapsed validation time: {toc-tic}s')
        print(f'Tr Loss: {loss_avg_tr[epoch]}, Val Loss: {loss_avg_vl[epoch]}')
            
        # save model if validation loss has decreased
        if loss_avg_vl[epoch] <= VALID_LOSS_MIN:
            print(f'The best model was saved!')
            torch.save(net, path_best_model)
            VALID_LOSS_MIN = loss_avg_vl[epoch]
            WAIT = 0
        # early stopping
        else:
            WAIT += 1
            if WAIT >= PATIENCE:
                print(f"Terminated training for early stopping at epoch {epoch+1}")
                break

    print(f'The last model was saved!')
    torch.save(net, path_last_model)

    # plot loss
    epochs_plot = range(1,(len(loss_avg_vl)+1))
    plt.plot(epochs_plot, loss_avg_tr)
    plt.plot(epochs_plot, loss_avg_vl)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.xticks(epochs_plot)
    plt.legend(('Training loss', 'Validation loss'), loc='upper right')
    plt.savefig('../losses_plot/MDE_AD_TVLosses_{}_{}_{}_{}_{}.pdf'.format(net.__class__.__name__, ordinal_method.__class__.__name__, tr_dataset.__class__.__name__, SELECTED_DISCRETIZER, N_CLASSES))
    plt.close()
    


    
################ TEST ################

# load model
if not os.path.exists(path_best_model):
    saved_model = torch.load(path_best_model, map_location=torch.device(device))
else:       
    saved_model = torch.load(path_best_model, map_location=torch.device(device))


print("[INFO] Testing the network...")
saved_model.eval() # set model to evaluation mode

metrics = [torchmetrics.MeanAbsoluteError().to(device), 
           torchmetrics.MeanSquaredError(squared=True).to(device), 
           metrics.absRel().to(device), 
           metrics.sqRel().to(device),
           metrics.RMSElog().to(device),
           metrics.Thresh().to(device)]


tic = time.time()
with torch.no_grad(): # turn off gradient tracking
    for image_ts, range_ts, range_disc_ts, range_disc_valid_ts in ts:
        image_ts = image_ts.to(device)
        range_ts = range_ts.to(device).flatten()
        range_disc_ts = range_disc_ts.to(device).flatten()
        range_disc_valid_ts = range_disc_valid_ts.to(device).flatten()
                
        pred_ts = saved_model(image_ts)
        pred_ts = pred_ts.permute(0, 2, 3, 1).flatten(0, 2)

        if LOSS_NAME == "MSE_continuous":
            range_ts_valid = range_ts[range_disc_valid_ts!=0]
            range_pred_ts_valid = pred_ts[range_disc_valid_ts!=0].squeeze()

        else:
            proba_ts = ordinal_method.to_probabilities(pred_ts)
            
            # without considering class 0 (invalid class)
            proba_ts_valid = proba_ts[range_disc_valid_ts!=0]
            range_ts_valid = range_ts[range_disc_valid_ts!=0]
            
            # from classes to depth
            range_pred_ts_valid = torch.sum(DEPTH_RANGES[None,:] * proba_ts_valid, 1)

        for metric in metrics:
            metric.update(range_pred_ts_valid, range_ts_valid)

toc = time.time()
print(f'Elapsed test time: {toc-tic}s')

# save metrics
with open('../metrics/MDE_AD_PMetrics_{}_{}_{}_{}_{}.csv'.format(net.__class__.__name__, ordinal_method.__class__.__name__, tr_dataset.__class__.__name__, SELECTED_DISCRETIZER, N_CLASSES),'w') as f:
    writer = csv.writer(f, dialect='excel')
    writer.writerow(["Metric", "Value"])
    for metric in metrics:
        if str(metric.__class__.__name__) == "Thresh":
            thres_metrics = metric.compute()
            for i, thre in enumerate(thres_metrics):
                writer.writerow([str(metric.__class__.__name__)+'_'+str(i), thre.item()])
        else:
            writer.writerow([str(metric.__class__.__name__), metric.compute().item()])