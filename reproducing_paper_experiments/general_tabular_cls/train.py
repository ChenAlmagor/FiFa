import torch
from torch import nn
from models import FiFa
from data import data_prep, DataSetCatCon

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, imputations_acc_justy
from augmentations import embed_data_mask
import matplotlib.pyplot as plt

import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='FiFa', type=str, choices=['FiFa'])
parser.add_argument('--act', default='gelu', type=str, choices=['relu', 'gelu', 'quad', 'half'])
parser.add_argument('--project_name', default='fifa', type=str)
parser.add_argument('--feat_type', default='all', type=str, choices=['all', 'num_only', 'cat_only'])
parser.add_argument('--final_dim_factor', default=1, type=float)
parser.add_argument('--num_dim', default=100, type=int, choices=[100, 2000])
parser.add_argument('--num_dim_factor', default=100, type=int)
parser.add_argument('--dropout', default=0.0, type=float, choices=[0.0, 0.1, 0.25, 0.5, 0.75])
parser.add_argument('--include_y', default=0, type=int, choices=[0, 1])
parser.add_argument('--dataset', default='1995_income', type=str,
                    choices=['1995_income', 'bank_marketing', 'qsar_bio', 'online_shoppers', 'blastchar', 'htru2',
                             'shrutime', 'spambase', 'philippine', 'loan_data', 'arcene', 'volkert', 'creditcard',
                             'arrhythmia', 'forest', 'kdd99'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--group_name', default=None, type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--train_mask_prob', default=0, type=float)

opt = parser.parse_args()
torch.manual_seed(opt.set_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.dataset in ['philippine']:
    opt.batchsize = 128
    opt.embedding_size = 8

print(f"Device is {device}.")

opt.include_y = bool(opt.include_y)

print('opt', opt)
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.dataset, opt.run_name)
os.makedirs(modelsave_path, exist_ok=True)

mask_params = {
    "mask_prob": opt.train_mask_prob,
    "avail_train_y": 0,
    "test_mask": opt.train_mask_prob
}

print('Downloading and processing the dataset, it might take some time.')
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(
    opt.dataset, opt.set_seed, mask_params, feat_type=opt.feat_type)
continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
if opt.dataset == 'volkert':
    y_dim = 10
else:
    y_dim = 2

train_bsize = opt.batchsize

train_ds = DataSetCatCon(X_train, y_train, cat_idxs, continuous_mean_std, is_pretraining=True)
trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, continuous_mean_std, is_pretraining=True)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs, continuous_mean_std, is_pretraining=True)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False)

cat_dims = np.append(np.array(cat_dims), np.array([y_dim])).astype(
    int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
print('opt', opt)
model = FiFa(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    y_dim=y_dim,
    final_dim_factor=opt.final_dim_factor,
    num_dim=opt.num_dim,
    dropout=opt.dropout,
    num_dim_factor=opt.num_dim_factor,
    include_y=opt.include_y,
    act=opt.act
)
vision_dset = False

if opt.dataset not in ["volkert"]:
    criterion = nn.BCEWithLogitsLoss().to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0

print('Training begins now. ', opt.model)

for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
        x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(
            device)

        # We are converting the data to embeddings in the next step
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

        logits = model.post_emb_factorization(x_categ_enc, x_cont_enc)

        if opt.dataset in ['volkert']:
            y = (x_categ[:, len(cat_dims) - 1]).long()
        else:
            y = (x_categ[:, len(cat_dims) - 1]).float()

        loss = criterion(logits.squeeze().float(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            if opt.dataset in ['volkert']:
                from utils import multiclass_acc_justy

                accuracy, auroc = multiclass_acc_justy(model, validloader, device)
                test_accuracy, test_auroc = multiclass_acc_justy(model, testloader, device)

                print('[EPOCH %d] VALID ACCURACY: %.3f' % (epoch + 1, accuracy))
                print('[EPOCH %d] TEST ACCURACY: %.3f' % (epoch + 1, test_accuracy))

                if accuracy > best_valid_accuracy:
                    best_valid_accuracy = accuracy
                    best_test_auroc = test_auroc
                    best_test_accuracy = test_accuracy
                    torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
            else:
                accuracy, auroc, val_loss = imputations_acc_justy(model, validloader, device, criterion=criterion)
                test_accuracy, test_auroc, test_loss = imputations_acc_justy(model, testloader, device,
                                                                             criterion=criterion)

                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, VALID LOSS: %.3f' %
                      (epoch + 1, accuracy, auroc, val_loss))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f, TEST LOSS: %.3f' %
                      (epoch + 1, test_accuracy, test_auroc, test_loss))

            if auroc > best_valid_auroc:
                best_valid_auroc = auroc
                best_test_auroc = test_auroc
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
        model.train()

total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
if opt.dataset not in ['volkert']:
    print('AUROC on best model:  %.3f' % (best_test_auroc))
else:
    print('Accuracy on best model:  %.3f' % (best_test_accuracy))


