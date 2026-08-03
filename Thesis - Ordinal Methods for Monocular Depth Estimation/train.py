import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('ordinal_method')
parser.add_argument('nclasses', type=int)
parser.add_argument('output')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--root', default='/data/depth')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--baseline', type=bool, default=False)
parser.add_argument('--spatial-reg', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import deep_ordinal
import data, models
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


########################## TRANSFORMATIONS ##########################

# TODO: change according the paper and the dataset
if args.dataset == 'KITTI' and args.model == 'SDNet':
    transforms = v2.Compose([
        v2.RandomCrop((352, 768)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(),
        v2.ToDtype(torch.float32, True)
    ])
elif args.dataset == 'KITTI':
    transforms = v2.Compose([
        v2.Resize((int(256*1.05), int(512*1.05))),
        v2.RandomCrop((256, 512)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.1, 0.1),
        v2.ToDtype(torch.float32, True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
elif args.dataset == 'Make3D':
    transforms = v2.Compose([
        v2.Resize(())
    ])
elif args.dataset == 'NYUDepthv2':
    transforms = v2.Compose([
        v2.Resize(())
    ])

########################## DATA loaders ##########################
def data_loader(args, split):

    if args.baseline:
        tr = getattr(data, args.dataset)(args.root, split, transforms, baseline=True)
    else:
        ds = getattr(data, args.dataset)(args.root, split, transforms)
        tr = ds = data.BinsDepth(ds, args.nclasses)

    if args.debug:
        tr = torch.utils.data.Subset(tr, range(100)) if split == 'val' else torch.utils.data.Subset(tr, range(1000))

    return torch.utils.data.DataLoader(tr, args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    
# Train set
tr_train = data_loader(args, 'train')

# Val set
tr_val = data_loader(args, 'val')

########################## MODEL ##########################

model = getattr(models, args.model)
if args.baseline:
    model = model(1, args.baseline).to(device)
    loss_fn = torch.nn.L1Loss()
else:
    ordinal = getattr(deep_ordinal, args.ordinal_method)(K=args.nclasses)
    model = model(ordinal.how_many_outputs(), args.baseline).to(device)
    model.ordinal = ordinal


if args.model == 'SDNet':
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0005, lr=0.0001)
else:
    optimizer = torch.optim.AdamW(model.parameters())

########################## TRAIN ##########################

def validate(model, val_loader, loss_metric, device, baseline):
    
    model.to(device).eval()
    avg_loss = 0.0

    with torch.no_grad():  # Disable gradient computation

        for images, depth, masks_valids, bins in val_loader:
            images, depth, masks_valids, bins = images.to(device), depth.to(device), masks_valids.to(device), bins.to(device)

            # Forward pass
            preds = model(images)
            preds = preds.permute(0, 2, 3, 1).to(device)

            # Calculate loss
            if baseline:
                # Convert into meters
                depth = depth.float().to(device) / 256.0
                
                if args.model == "SDNet":
                    # Depth interval for SDNet
                    depth = torch.clamp(depth, 2., 80.) 

                # Flatten
                preds, depth, masks_valids = preds.flatten(), depth.flatten(), masks_valids.flatten()
                # Loss calculation
                loss = (masks_valids * loss_metric(preds, depth)).sum() / masks_valids.sum()

            else:
                # Loss calculation
                preds, bins, masks_valids = preds.flatten(), bins.flatten(), masks_valids.flatten()
                loss = (masks_valids * loss_metric.compute_loss(preds, bins)).sum() / masks_valids.sum()
            
            # Accumulate loss for each batch
            avg_loss += float(loss)
        
        # Avg loss
        avg_loss /= len(val_loader)
    return avg_loss


best_val_loss = float('inf')

for epoch in range(args.epochs):

    model.to(device).train()
    tic = time()
    tot_loss = 0.0

    for images, depth, masks_valids, bins in tr_train:
        images, depth, masks_valids, bins = images.to(device), depth.to(device), masks_valids.to(device), bins.to(device)
        
        # Forward pass
        preds = model(images)
        preds = preds.permute(0, 2, 3, 1).to(device)

        # https://arxiv.org/pdf/2407.20959
        # TODO spatial regularization
        # loss += 0.1*CSNP(preds, preds.shape[1])
        # Compute the loss
        # If the model is the baseline, bins are replaced for the ground truth
        if args.baseline:
            # Convert into meters
            depth = depth.float().to(device) / 256.0
            
            if args.model == "SDNet":
                # Depth interval for SDNet
                depth = torch.clamp(depth, 2., 80.)
            
            # Flatten after processing
            preds, depth, masks_valids = preds.flatten(), depth.flatten(), masks_valids.flatten()
            # Loss calculation
            loss = (masks_valids * loss_fn(preds, depth)).sum() / masks_valids.sum()

        else:
            # Loss calculation
            preds, bins, masks_valids = preds.flatten(), bins.flatten(), masks_valids.flatten()
            loss = (masks_valids * ordinal.compute_loss(preds, bins)).sum() / masks_valids.sum()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        tot_loss += float(loss)

    # Average loos
    avg_loss = tot_loss / len(tr_train)  
    
    # Validate the model
    val_loss = validate(model, tr_val, loss_fn if args.baseline else ordinal, device, args.baseline)
    

    toc = time()

    # Saving the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.cpu(), args.output)
        print("Saved best model.")

    # print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss}')
    print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {toc-tic:.0f}s")
    

# torch.save(model.cpu(), args.output)

