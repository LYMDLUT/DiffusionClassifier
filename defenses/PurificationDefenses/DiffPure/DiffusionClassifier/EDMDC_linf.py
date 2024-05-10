import math
from typing import Tuple, Any

import numpy as np
import torch.cuda
from scipy import integrate
from torch import nn
from torch.autograd import Function

from .DiffusionClassifierBase import DiffusionClassifierSingleHeadBase
from .utils import *


class EDMEulerIntegralDC(DiffusionClassifierSingleHeadBase):
    def __init__(
        self,
        unet: nn.Module,
        timesteps=torch.linspace(1e-4, 3, 1001),
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        *args,
        **kwargs,
    ):
        super(EDMEulerIntegralDC, self).__init__(unet, *args, **kwargs)
        self.unet = unet
        self.reset_param(P_mean, P_std, timesteps, sigma_data)
        self._init()
        self.noise = torch.randn(1001, 3, 32, 32, device=torch.device("cuda:0"), dtype=self.precision)

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

    @torch.no_grad()
    def unet_loss_without_grad(
        self, x: Tensor, y: int or Tensor = None, batch_size=128, generator: torch.Generator = torch.default_generator
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
            #noise = self.noise[count : count + B].to(now_x.device)
            noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device, dtype=self.precision)
            noised_x = now_x + noise * sigma.view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma, now_y)
            loss = torch.sum(weight * torch.mean((pre - now_x) ** 2, dim=[1, 2, 3]))
            result = result + loss
            count += B
        return result

    @torch.enable_grad()
    def unet_loss_with_grad(
        self,
        x: Tensor,
        y: int or Tensor = None,
        batch_size=128,
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
            #noise = self.noise[count : count + B].to(now_x.device)
            noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device)
            noised_x = now_x + noise * sigma.view(-1, 1, 1, 1)

            pre = self.unet(noised_x, sigma, now_y)
            loss = torch.sum(weight * torch.mean((pre - now_x) ** 2, dim=[1, 2, 3]))
            loss.backward()
            result = result + loss
            count += B
        return result

    def one_step_denoise(self, x: Tensor, normalize=True, sigma=0.5, y=None) -> Tensor:
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2 if normalize else x
        x0 = self.unet(x, torch.zeros((x.shape[0],), device=x.device) + sigma, y)
        x0 = x0 / 2 + 0.5 if normalize else x0
        return x0


class EDMEulerIntegralFunction(Function):
    """
    batchsize should be 1
    """

    classifier = None

    @staticmethod
    def forward(ctx: Any, *args, **kwargs):
        x = args[0]
        target_class_tensor = EDMEulerIntegralFunction.classifier.target_class
        ctx.target_class_tensor = target_class_tensor
        assert x.shape[0] == 1, "batch size should be 1"
        x = x.detach()  # because we will do some attribute modification
        x.requires_grad = True
        logit = []
        dlogit_dx = []
        for class_id in EDMEulerIntegralFunction.classifier.target_class:
            x.grad = None
            with torch.enable_grad():
                logit.append(EDMEulerIntegralFunction.classifier.unet_loss_with_grad(x, class_id))
                grad = x.grad.clone()
                dlogit_dx.append(grad)
            x.grad = None
        logit = torch.tensor(logit, device=torch.device("cuda")).unsqueeze(0)  # 1, num_classes
        logit = logit * -1
        ctx.dlogit_dx = [i * -1 for i in dlogit_dx]
        result = logit
        return result

    @staticmethod
    def backward(ctx: Any, grad_logit, lower_bound=False):
        """
        :param ctx:
        :param grad_logit: 1, num_classes
        :return:
        """
        dlogit_dx = ctx.dlogit_dx
        dlogit_dx = torch.stack(dlogit_dx)  # num_classes, *x_shape
        dlogit_dx = dlogit_dx.permute(1, 2, 3, 4, 0)  # *x_shape, num_classes
        grad = dlogit_dx @ grad_logit.squeeze()
        return grad


class EDMEulerIntegralWraped(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(EDMEulerIntegralWraped, self).__init__()
        EDMEulerIntegralFunction.classifier = EDMEulerIntegralDC(*args, **kwargs)
        self.unet = EDMEulerIntegralFunction.classifier.unet
        self.knnclassifier = EDMEulerIntegralFunction.classifier
        self.eval().to(torch.device("cuda")).requires_grad_(False)

    def forward(self, x):
        if x.requires_grad is False:  # eval mode, prediction
            logit = self.knnclassifier.forward(x)
        else:
            # crafting adversarial patches, requires_grad mode
            logit = EDMEulerIntegralFunction.apply(x)
        return logit


class EDMGaussQuadratureDC(DiffusionClassifierSingleHeadBase):
    def __init__(
        self,
        unet: nn.Module,
        t0=1e-4,
        t1=3,
        quadrature_order=1000,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        num_classes=10,
        *args,
        **kwargs,
    ):
        super(EDMGaussQuadratureDC, self).__init__(unet, *args, **kwargs)
        self.unet = unet
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.t0, self.t1 = t0, t1
        self.quadrature_order = quadrature_order
        self.num_classes = num_classes

    def log_normal_distribution_pdf(self, x: Tensor) -> Tensor:
        return (
            1
            / (x * self.P_std * math.sqrt(2 * math.pi))
            * torch.exp(-((torch.log(x) - self.P_mean) ** 2) / 2 * self.P_std**2)
        )

    def p_t_l_x_t(self, t: np.array, x: Tensor, y: Tensor = None) -> np.array:
        """
        :param t:  (quadrature_order, )
        :param x:
        :param y:
        :return: (quadrature_order, )
        """
        t = torch.from_numpy(t).to(self.device)
        p_t = self.log_normal_distribution_pdf(t)

        # calculate
        weight = (t**2 + self.sigma_data**2) / (t * self.sigma_data) ** 2 * p_t
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        y = y.repeat(t.numel()) if y is not None else None
        x = self.transform(x)
        x = x.repeat(t.numel(), 1, 1, 1)
        noise = torch.randn_like(x)
        noised_x = x + noise * t.view(-1, 1, 1, 1)

        pre = self.unet(noised_x, t, y)
        loss = weight * torch.mean((pre - x) ** 2, dim=[1, 2, 3])
        return loss.cpu().numpy()

    @torch.no_grad()
    def unet_loss_without_grad(
        self,
        x: Tensor,
        y: int or Tensor = None,
    ) -> float:
        """
        Calculate the diffusion loss, \int p(t) l(x, t) dt
        :param x: in range [0, 1]
        :param y: int or Tensor
        """
        result = integrate.fixed_quad(lambda t: self.p_t_l_x_t(t, x, y), self.t0, self.t1, n=self.quadrature_order)
        return result[0]


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

