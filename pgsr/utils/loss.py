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
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = F.pad(grad_img[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()
    return grad_img
