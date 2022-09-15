import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from augmentations import embed_data_mask
import torch.nn as nn
import sklearn
def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  


def imputations_acc_justy(model,dloader,device, criterion ):

        model.eval()
        m = nn.Sigmoid()
        y_test = torch.empty(0).to(device)
        y_pred = torch.empty(0).to(device)
        prob = torch.empty(0).to(device)
        with torch.no_grad():
            for i, data in enumerate(dloader, 0):
                x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)

                logits = model.post_emb_factorization(x_categ_enc, x_cont_enc)
                logits = logits.squeeze()
                y = x_categ[:,-1].float()
                y_test = torch.cat([y_test,y],dim=0)
                sig_logits = m(logits)
                y_pred = torch.cat([y_pred, sig_logits > 0.5],dim=0)
                prob = torch.cat([prob,logits],dim=0)

        correct_results_sum = (y_pred == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]*100
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())


        loss = criterion(prob.squeeze().float(), y_test)
        return acc, auc, loss





def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = False
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            logits = model.post_emb_factorization(x_categ_enc, x_cont_enc)
            logits = logits.squeeze()
            y = x_categ[:, -1].long()
            y_test = torch.cat([y_test, y], dim=0)


            y_pred = torch.cat([y_pred,torch.argmax(m(logits), dim=1).float()],dim=0)

    acc = sklearn.metrics.balanced_accuracy_score(y_test.cpu(), y_pred.cpu())

    return acc, 0,