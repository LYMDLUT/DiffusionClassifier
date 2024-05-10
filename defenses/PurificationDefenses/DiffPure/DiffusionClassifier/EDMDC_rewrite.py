import math
from typing import Tuple, Any

from torch import nn
from .utils import *

class EDMEulerIntegralDC(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        timesteps=torch.linspace(0.5, 1, 30),
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        *args,
        **kwargs,
    ):
        super(EDMEulerIntegralDC, self).__init__(*args, **kwargs)
        self.device = torch.device("cuda:0")
        self.unet = unet
        self.reset_param(P_mean, P_std, timesteps, sigma_data)
        self.transform = lambda x: (x - 0.5) * 2
        self.noise = torch.randn(1001, 3, 32, 32, device=torch.device("cuda:0"))
        
        
    def reset_param(self, P_mean=-1.2, P_std=1.2, timesteps=torch.linspace(1e-4, 3, 1001), sigma_data=0.5):
        timesteps = timesteps.to(self.device)
        self.timesteps = timesteps[:-1]
        self.dt = timesteps[1:] - timesteps[:-1]

        # storing
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        x = self.timesteps
        self.p_x = 1 / (x * P_std * math.sqrt(2 * math.pi)) * torch.exp(-((torch.log(x) - P_mean) ** 2) / 2 * P_std**2)

    def unet_loss_with_grad(
        self,
        x: Tensor,
        y: int or Tensor = None,
        batch_size=32,
        generator: torch.Generator = torch.cuda.default_generators[0],
    ) -> Tensor:
        """
        Calculate the diffusion loss
        x should be in range [0, 1]
        """
        result = 0
        count = 0
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        for sigma in self.timesteps.split(batch_size, dim=0):
            B = sigma.shape[0]
            weight = (
                (sigma**2 + self.sigma_data**2)
                / (sigma * self.sigma_data) ** 2
                * self.p_x[count : count + B]
                * self.dt[count : count + B]
            )
            now_y = y.repeat(sigma.numel()) if y is not None else None
            now_x = self.transform(x)
            now_x = now_x.repeat(sigma.numel(), 1, 1, 1)
            noise = self.noise[count : count + B].to(now_x.device)
            # noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device)
            noised_x = now_x + noise * sigma.view(-1, 1, 1, 1)

            pre = self.unet(noised_x, sigma, now_y)
            loss = torch.sum(weight * torch.mean((pre - now_x) ** 2, dim=[1, 2, 3]))
            result = result + loss
            count += B
        return result


class EDMEulerIntegralWraped(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(EDMEulerIntegralWraped, self).__init__()
        self.classifier = EDMEulerIntegralDC(*args, **kwargs)

    def forward(self, x):
        logit = torch.zeros(10).to(x.device)
        for class_id in [0,1,2,3,4,5,6,7,8,9]:
            with torch.enable_grad():
                logit[class_id] = self.classifier.unet_loss_with_grad(x, class_id)
        logit = logit.unsqueeze(0)  # 1, num_classes
        logit = logit * -1
        return logit


import torch
from torch import nn
import math
from typing import Tuple, Any
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from models.unets import get_edm_cifar_cond


if __name__ == "__main__":
    unet = get_edm_cifar_cond(use_fp16=False).cuda()
    unet.requires_grad=False
    dc = EDMEulerIntegralWraped(unet=unet)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5]),
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test.data = cifar10_test.data[:32]
    cifar10_test.targets = cifar10_test.targets[:32]

    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=False, num_workers=5, batch_size=1, drop_last=True)

    cls_init_acc_list = []
    pure_acc_list = []
    epsilon = torch.tensor(8 / 255.).cuda()
    alpha =  torch.tensor(1 / 255.).cuda()
    upper_limit = torch.tensor(1).cuda()
    lower_limit = torch.tensor(0).cuda()

    def clamp1(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    for sample_image, y_val in cifar10_test_loader:
        sample_image = sample_image.cuda()
        y_val = y_val.cuda()
        # with torch.no_grad():
        #     yy_init= dc(sample_image)
        #     cls_acc_init = sum((yy_init.argmax(dim=-1) == y_val)) / y_val.shape[0]
        #     print("init_cls_acc:",cls_acc_init)
        #     cls_init_acc_list.append(cls_acc_init.cpu().item())
        delta = torch.zeros_like(sample_image)
        delta.requires_grad = True
        for _ in tqdm(range(20), colour='red'):
            eot = 0
            for _ in range(1):
                tmp_out = dc(sample_image + delta)
                loss = F.cross_entropy(tmp_out, y_val)
                loss.backward()
                grad = delta.grad.detach()
                eot += grad
                delta.grad.zero_()
            d = clamp1(delta + alpha * eot.sign(), -epsilon, epsilon)
            d = clamp1(d, lower_limit - sample_image, upper_limit - sample_image)
            delta.data = d
        adv_out = (sample_image + delta)
        with torch.no_grad():    
            yy= dc(adv_out)
            pure_cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
            print("pure_cls_acc:", str(pure_cls_acc))
            pure_acc_list.append(pure_cls_acc.cpu().item())
    print(sum(pure_acc_list)/len(pure_acc_list))
    # print(sum(cls_init_acc_list)/len(cls_init_acc_list))

