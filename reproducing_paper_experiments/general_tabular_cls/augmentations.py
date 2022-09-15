import torch
import numpy as np


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape

    x_cont_enc = torch.empty(n1, n2, model.dim)
    for i in range(model.num_continuous):
        x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    # ######REMOVE!!!!!#####
    # if vision_dset:
    #     pos = np.tile(np.arange(x_categ.shape[-1]), (x_categ.shape[0], 1))
    #     pos = torch.from_numpy(pos).to(device)
    #     pos_enc = model.pos_encodings(pos)
    #     x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc