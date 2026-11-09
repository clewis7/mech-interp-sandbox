import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet
from torchvision.models.googlenet import BasicConv2d

N_CLASSES = 100
IMG_SIZE = 96  # 32px CIFAR upsampled; both nets end at a 3x3 map
DEVICE = "cuda"
DATA_PATH = "data/cifar100.pt"
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


# ---- "old view": InceptionV1 as in 2014 (no BatchNorm, no residuals, no LayerNorm) ----
def make_inception_v1():
    model = googlenet(weights=None, aux_logits=False, init_weights=False, num_classes=N_CLASSES)
    for m in model.modules():
        if isinstance(m, BasicConv2d):
            m.bn = nn.Identity()  # torchvision added BN; the original had none
            m.conv.bias = nn.Parameter(torch.zeros(m.conv.out_channels))  # convs were bias=False because of BN
            nn.init.kaiming_normal_(m.conv.weight, nonlinearity="relu")  # default std=0.01 init vanishes without BN
    return model


def make_convnext():
    return timm.create_model("convnext_femto", pretrained=False, num_classes=N_CLASSES,
                             dims=(52, 104, 208, 416), ls_init_value=1e-2, drop_path_rate=0.1)


# ---- hook targets: last-layer neurons, pre-nonlinearity ----
# InceptionV1 inception5b: final conv of each branch, before ReLU (384 + 384 + 128 + 128 = 1024)
def inception_hook_modules(model):
    b = model.inception5b
    return [b.branch1.bn, b.branch2[1].bn, b.branch3[1].bn, b.branch4[1].bn]


# ConvNeXt last block MLP hidden, before GELU (4 * 384 = 1536)
def convnext_hook_module(model):
    return model.stages[-1].blocks[-1].mlp.fc1


acts = {}


def save_to(key):
    def hook(module, inp, out):
        acts[key] = out.detach().clone()
    return hook


# ---- smoke test ----
if __name__ == "__main__":
    inception = make_inception_v1().to(DEVICE)
    convnext = make_convnext().to(DEVICE)

    print(f"InceptionV1 params: {sum(p.numel() for p in inception.parameters()) / 1e6:.2f}M")
    print(f"ConvNeXt    params: {sum(p.numel() for p in convnext.parameters()) / 1e6:.2f}M")

    for i, m in enumerate(inception_hook_modules(inception)):
        m.register_forward_hook(save_to(f"inc_{i}"))
    fc1 = convnext_hook_module(convnext)
    fc1.register_forward_hook(save_to("cnx"))
    channels_last = isinstance(fc1, nn.Linear)  # timm femto may use a 1x1 conv instead

    # real probe images, preprocessed on GPU
    data = torch.load(DATA_PATH)
    imgs = data["test_imgs"][data["probe_idx"][:8]].to(DEVICE)
    x = imgs.float() / 255
    x = F.interpolate(x, size=IMG_SIZE, mode="bilinear", align_corners=False)
    x = (x - MEAN.to(DEVICE)) / STD.to(DEVICE)

    with torch.no_grad():
        inception.eval()
        convnext.eval()
        out_inc = inception(x)
        out_cnx = convnext(x)

    inc_acts = torch.cat([acts[f"inc_{i}"] for i in range(4)], dim=1)  # (B, 1024, H, W)
    cnx_acts = acts["cnx"].permute(0, 3, 1, 2) if channels_last else acts["cnx"]  # (B, 1536, H, W)

    print("logits:", tuple(out_inc.shape), tuple(out_cnx.shape))
    print("inception5b pre-ReLU:", tuple(inc_acts.shape), "min", inc_acts.min().item())
    print("convnext fc1 pre-GELU:", tuple(cnx_acts.shape), "min", cnx_acts.min().item())
    print("per-neuron spatial max:", tuple(inc_acts.amax(dim=(2, 3)).shape), tuple(cnx_acts.amax(dim=(2, 3)).shape))