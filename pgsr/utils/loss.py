import numpy as np
import torch
import torch.nn.functional as F


# Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/utils/loss_utils.py#L92-L104
def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd, 1:wd-1]
    top_point = img[..., 0:hd-2, 1:wd-1]
    right_point = img[..., 1:hd-1, 2:wd]
    left_point = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_min = grad_img.min()
    grad_range = (grad_img.max() - grad_min).clamp_min(torch.finfo(grad_img.dtype).eps)
    grad_img = (grad_img - grad_min) / grad_range
    grad_img = F.pad(grad_img[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()
    return grad_img


# Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/utils/loss_utils.py#L106-L141
def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask
