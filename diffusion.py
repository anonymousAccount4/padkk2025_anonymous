from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema import EMA
from model import UnetModel


def extract(a, t, x_shape):
    b = t.shape
    out = a.gather(-1, t)
    return out.reshape(b[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        img_channels,
        betas,
        mean_type="eps",
        var_type="fixedlarge",
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=1,
        ema_update_rate=1,
        eta=0,
        cond="none",
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.register_buffer("betas", partial(torch.tensor, dtype=torch.float32)(betas))
        self.step = 0

        self.cond = cond
        self.img_channels = img_channels

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[: self.num_timesteps]

        # calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_bar)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1.0 / alphas_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_bar", torch.sqrt(1.0 / alphas_bar - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
        )

    @torch.no_grad()
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        b, l, f = x_0.shape
        x_0 = x_0.reshape(b * l, -1)
        x_t = x_t.reshape(b * l, -1)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    @torch.no_grad()
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    @torch.no_grad()
    def p_mean_variance(self, x_t, t, c=None, use_ema=False):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            "fixedlarge": torch.log(
                torch.cat([self.posterior_var[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape).reshape(-1, 1)
        # Mean parameterization
        if self.mean_type == "xprev":  # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, c)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == "xstart":  # the model predicts x_0
            x_0 = self.model(x_t, t, c)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == "eps":  # the model predicts epsilon
            eps = self.model(x_t, t, c)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var

    @torch.no_grad()
    def remove_noise_ddim(self, x, t, c=None, use_ema=False):
        if use_ema:
            e = self.ema_model(x, t, c)

        else:
            e = self.model(x, t, c)

        return (
            extract(self.reciprocal_sqrt_alphas, t, x.shape) * x
            - extract(
                self.sqrt_one_minus_alphas_cumprod * self.reciprocal_sqrt_alphas,
                t,
                x.shape,
            )
            * e
            + extract((1 - self.alphas_cumprod_prev - self.sigma**2).sqrt(), t, x.shape)
            * e
        )

    @torch.no_grad()
    def sample(self, x, mot, text, dynamic_timer, use_ema=False):
        if self.cond == "motion":
            c = mot
        elif self.cond == "text":
            c = text
        elif self.cond == "none":
            c = None

        device = x.device
        b, l, f = x.shape

        t, _ = dynamic_timer.search(x.reshape(-1, f).cpu().detach().numpy(), 1)
        t = t.reshape(b, l)
        t = np.mean(t, axis=1).repeat(l)
        t = torch.Tensor(t).to(device).to(torch.int64)
        t = torch.sqrt(t).to(torch.int64)*3 + 30

        x = x.reshape(-1, f)
        noise = torch.randn_like(x)
        x_noise = self.perturb_x(x, t, noise).reshape(b, l, f)
        for _ in range(t.max().item() - 1, -1, -1):
            cur = t == 1
            t[t > 0] -= 1
            mean, log_var = self.p_mean_variance(x_t=x_noise, t=t, c=c, use_ema=use_ema)
            # if t > 0:
            #     noise = torch.randn_like(x_noise)
            # else:
            #     noise = 0
            noise = torch.randn_like(x_noise).reshape(b * l, -1)
            noise[t == 0] = 0
            x_noise_ = mean + torch.exp(0.5 * log_var) * noise
            x_noise = x_noise.reshape(b * l, -1)
            x_noise[(t > 0) | cur] = x_noise_[(t > 0) | cur]
            x_noise = x_noise.reshape(b, l, -1)

        return x_noise.reshape(-1, f)

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=False):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels).to(device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.betas.sqrt(), t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, c=None):
        b, l, f = x.shape

        x = x.reshape(-1, f)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise).reshape(b, l, f)
        estimated_noise = self.model(perturbed_x, t, c)

        if self.mean_type == "xstart":  # the model predicts x_0
            target = x
        elif self.mean_type == "eps":  # the model predicts epsilon
            target = noise

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise.reshape(-1, f), target.reshape(-1, f))
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise.reshape(-1, f), target.reshape(-1, f))

        return loss

    def forward(self, x, mot, text):
        if self.cond == "motion":
            c = mot
        elif self.cond == "text":
            c = text
        elif self.cond == "none":
            c = None

        b, t, f = x.shape
        device = x.device

        t = torch.randint(0, self.num_timesteps, (b * t,), device=device)

        return self.get_losses(x, t, c)


class MDVAD(nn.Module):
    def __init__(
        self,
        vis_channel,
        mot_channel,
        text_channel,
        betas,
        ch_mult,
        num_res_blocks,
        motion_pretrained,
        text_pretrained,
        use_ema=False,
        mean_type="eps",
        var_type="fixedlarge",
        loss_type="l2",
        mode="train",
    ):
        super().__init__()
        self.use_ema = use_ema
        self.vis_channel = vis_channel
        self.mot_channel = mot_channel
        self.text_channel = text_channel
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        self.motion_pretrained = motion_pretrained
        self.text_pretrained = text_pretrained

        self.register_buffer("betas", partial(torch.tensor, dtype=torch.float32)(betas))

        if mode == "train":
            self.mot_model = self.load_state_model(self.mot_channel)
            self.text_model = self.load_state_model(self.text_channel)

        elif mode == "test":
            self.mot_model = self.load_state_model(
                self.mot_channel, self.motion_pretrained
            )
            self.text_model = self.load_state_model(
                self.text_channel, self.text_pretrained
            )

        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[: self.num_timesteps]

        # calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_bar)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_betas", torch.sqrt(self.betas))
        self.register_buffer("sqrt_one_minus_betas", torch.sqrt(1.0 - self.betas))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
        )

    def load_state_model(self, cond_dim, pretrained=None):
        model = UnetModel(
            in_channel=self.vis_channel,
            cond_dim=cond_dim,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
        )
        if pretrained is not None:
            diffusion = GaussianDiffusion(
                model,
                self.vis_channel,
                self.betas.detach().numpy(),
                mean_type=self.mean_type,
                var_type=self.var_type,
                loss_type=self.loss_type,
            )
            diffusion.load_state_dict(torch.load(pretrained))
            diffusion.eval()
            model = diffusion.model
        return model

    @torch.no_grad()
    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        b, l, f = x_0.shape
        x_0 = x_0.reshape(b * l, -1)
        x_t = x_t.reshape(b * l, -1)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    @torch.no_grad()
    def p_mean_variance(self, model, x_t, t, c=None, use_ema=True):
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            "fixedlarge": torch.log(
                torch.cat([self.posterior_var[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape).reshape(-1, 1)

        x_0 = model(x_t, t, c)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)

        return model_mean, model_log_var

    @torch.no_grad()
    def perturb(self, x, t, noise):
        return (
            extract(self.sqrt_one_minus_betas, t, x.shape) * x
            + extract(self.sqrt_betas, t, x.shape) * noise
        )

    @torch.no_grad()
    def perturb_prod(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    @torch.no_grad()
    def forward_step(self, x_noise, t, device="cpu"):
        t_batch = (
            x_noise.new_ones(
                [
                    x_noise.shape[0],
                ],
                dtype=torch.long,
            ).to(device)
            * t
        )
        return self.perturb(x_noise, t_batch, torch.randn_like(x_noise))

    @torch.no_grad()
    def reverse_step(self, model, x_noise, c, t, shape, device="cpu"):
        b, l = shape
        cur = t == 1
        t[t > 0] -= 1
        mean, log_var = self.p_mean_variance(
            model=model, x_t=x_noise, t=t, c=c, use_ema=self.use_ema
        )
        noise = torch.randn_like(x_noise).reshape(b * l, -1)
        noise[t == 0] = 0
        x_noise_ = mean + torch.exp(0.5 * log_var) * noise
        x_noise = x_noise.reshape(b * l, -1)
        x_noise[(t > 0) | cur] = x_noise_[(t > 0) | cur]
        x_noise = x_noise.reshape(b, l, -1)

        return x_noise

    @torch.no_grad()
    def extract_att(self, x, text, time):
        device = x.device
        x = x.repeat(1, 8, 1)
        t_batch = (
            x.new_ones(
                [
                    x.shape[0],
                ],
                dtype=torch.long,
            ).to(device)
            * time
        )
        x_noise = self.perturb_prod(x, t_batch, torch.randn_like(x))
        times = generate_adaptive_jump_schedule(time, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        for t_last, t_cur in time_pairs:
            t_batch = (
                x_noise.new_ones(
                    [
                        x_noise.shape[0],
                    ],
                    dtype=torch.long,
                ).to(device)
                * t_last
            )
            text_h, text_att, text_feats = self.text_model.extract_att(
                x_noise, t_batch, text
            )
        #     mot_h, mot_att, mot_feats = self.mot_model.extract_att(x_noise, t_batch, mot)
        # return mot_h, mot_att, mot_feats, text_h, text_att, text_feats
        return text_h, text_att, text_feats

    @torch.no_grad()
    def sample_diffusion_sequence(self, x, mot, text, time):
        device = x.device
        t_batch = (
            x.new_ones(
                [
                    x.shape[0],
                ],
                dtype=torch.long,
            ).to(device)
            * time
        )
        x_noise = self.perturb_prod(x, t_batch, torch.randn_like(x))
        times = generate_adaptive_jump_schedule(time, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        sequence = [x_noise.reshape(-1, 1024)]
        for t_last, t_cur in time_pairs:
            if t_last > t_cur and t_last % 2 == 1:
                x_noise = self.reverse_step(
                    self.text_model, x_noise, text, t_last, device
                )
            elif t_last > t_cur and t_last % 2 == 0:
                x_noise = self.reverse_step(
                    self.mot_model, x_noise, mot, t_last, device
                )
            else:
                x_noise = self.forward_step(x_noise, t_cur, device)
            sequence.append(x_noise.reshape(-1, 1024))
        return sequence

    @torch.no_grad()
    def sample(self, x, mot, text, dynamic_timer, n):
        device = x.device
        b, l, f = x.shape

        t, _ = dynamic_timer.search(
            x.reshape(-1, x.shape[-1]).cpu().detach().numpy(), 1
        )
        t = np.mean(t, axis=1)
        t = t.reshape(b, l)
        t = np.mean(t, axis=1).repeat(l)
        t = torch.Tensor(t).to(device).to(torch.int64)

        t = torch.sqrt(t).to(torch.int64) * 3 + 30

        t[t % 2 == 1] += 1

        x = x.reshape(-1, f)
        x_noise = self.perturb_prod(x, t, torch.randn_like(x)).reshape(b, l, f)
        for _ in range(t.max().item() - 1, -1, -1):
            x_noise = self.reverse_step(self.text_model, x_noise, text, t, (b, l), device)
            x_noise = self.reverse_step(self.mot_model, x_noise, mot, t, (b, l), device)

        return x_noise.reshape(-1, 1024)

    def get_losses(self, x, t, mot, text):
        b, l, f = x.shape

        x = x.reshape(-1, f)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_prod(x, t, noise).reshape(b, l, f)

        mot_estimated_noise = self.mot_model(perturbed_x, t, mot)
        text_estimated_noise = self.text_model(perturbed_x, t, text)

        if self.mean_type == "xstart":  # the model predicts x_0
            target = x
        elif self.mean_type == "eps":  # the model predicts epsilon
            target = noise

        mot_loss = F.mse_loss(
            mot_estimated_noise.reshape(-1, 1024), target.reshape(-1, 1024)
        )
        text_loss = F.mse_loss(
            text_estimated_noise.reshape(-1, 1024), target.reshape(-1, 1024)
        )

        return mot_loss + text_loss

    def forward(self, x, mot, text):

        b, t, f = x.shape
        device = x.device

        t = torch.randint(0, self.num_timesteps, (b * t,), device=device)
        # t = torch.randint(0, 100, (b * t,), device=device)
        return self.get_losses(x, t, mot, text)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def generate_adaptive_jump_schedule(T, jump_length, jump_n_sample):
    t = T
    ts = []

    jumps = {}
    for j in range(0, T, jump_length):
        jumps[j] = int(jump_n_sample - 1)

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)
    return ts
