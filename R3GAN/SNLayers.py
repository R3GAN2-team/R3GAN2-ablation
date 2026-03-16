import torch
import torch.nn as nn
import math

def _l2_normalize(v):
    return v / (v.norm(dim=0, keepdim=True))

@torch.no_grad()
def power_iteration_sigma(Wmat, u, n_steps=1):
    # Ensure float32 for stability.
    W = Wmat.to(torch.float32)
    u_new = u.to(torch.float32)

    for _ in range(n_steps):
        v_new = _l2_normalize(W.t() @ u_new)
        u_new = _l2_normalize(W @ v_new)

    sigma = (u_new.t() @ (W @ v_new)).squeeze()  # scalar
    return sigma, u_new, v_new

class SpectralNormalizedWeight(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, KernelSize, Centered, power_iters=10):
        super().__init__()
        self.Centered = Centered
        self.power_iters = power_iters

        self.Weight = nn.Parameter(torch.randn(OutputChannels, InputChannels // Groups, *KernelSize))

        # Buffers for power iteration (top singular vectors of Wmat).
        out_ch = OutputChannels
        in_ch = (InputChannels // Groups) * int(torch.tensor(KernelSize).prod().item())
        u0 = torch.randn(out_ch, 1)
        u0 = _l2_normalize(u0)
        self.register_buffer("u", u0)  # updated in NormalizeWeight()

    def _center(self, w):
        # Center over in+spatial dims (all dims except output channel dim).
        if not self.Centered:
            return w
        dims = list(range(1, w.ndim))
        return w - torch.mean(w, dim=dims, keepdim=True)

    def _reshape_to_mat(self, w):
        # Standard SN reshape: [out, in/groups*kH*kW]
        return w.reshape(w.shape[0], -1)

    def Evaluate(self, w, update_u):
        w = self._center(w)
        Wmat = self._reshape_to_mat(w)

        fan_in = Wmat.shape[1]
        fan_out = Wmat.shape[0]
        target_sigma = math.sqrt(fan_in) +  math.sqrt(fan_out)

        # Estimate sigma_max(Wmat) via power iteration using stored u.
        sigma, u_new, _v_new = power_iteration_sigma(Wmat, self.u, n_steps=self.power_iters)

        if update_u:
            self.u.copy_(u_new.to(self.u.dtype))

        denom = (sigma / target_sigma)
        w_sn = w / denom
        return w_sn

    def forward(self):
        # IMPORTANT: do not update u during forward to keep multiple forward passes consistent.
        return self.Evaluate(self.Weight.to(torch.float32), update_u=False)

    def NormalizeWeight(self):
        # Called after each optimizer step (forced normalization + update u).
        w_new = self.Evaluate(self.Weight.detach().to(torch.float32), update_u=True)
        self.Weight.copy_(w_new.to(self.Weight.dtype))

class WeightNormalizedConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, EnablePadding, KernelSize, Centered):
        super(WeightNormalizedConvolution, self).__init__()
        
        self.Groups = Groups
        self.EnablePadding = EnablePadding
        self.Weight = SpectralNormalizedWeight(InputChannels, OutputChannels, Groups, KernelSize, Centered)

    def forward(self, x, Gain=1):
        w = self.Weight()

        Wmat = w.reshape(w.shape[0], -1)
        fan_in = Wmat.shape[1]
        fan_out = Wmat.shape[0]
        target_sigma = math.sqrt(fan_in) +  math.sqrt(fan_out)

        w = w * (Gain / target_sigma)
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        return nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,) if self.EnablePadding else 0, groups=self.Groups)
        
def Convolution(InputChannels, OutputChannels, KernelSize, Groups=1, Centered=False):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize], Centered)

class BiasedPointwiseConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Centered=False):
        super(BiasedPointwiseConvolution, self).__init__()
        
        self.Weight = SpectralNormalizedWeight(InputChannels + 1, OutputChannels, 1, [1, 1], Centered)
        
    def forward(self, x, Gain=1):
        w = self.Weight()

        Wmat = w.reshape(w.shape[0], -1)
        fan_in = Wmat.shape[1]
        fan_out = Wmat.shape[0]
        target_sigma = math.sqrt(fan_in) +  math.sqrt(fan_out)

        w = w / target_sigma
        b = w[:, -1, :, :].view(-1)
        w = w[:, :-1, :, :] * Gain
        
        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype))