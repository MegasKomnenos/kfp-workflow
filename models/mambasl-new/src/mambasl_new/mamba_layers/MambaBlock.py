import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class Mamba_TimeVariant(nn.Module):
    def __init__(
        self,
        d_model,
        d_input=None,
        d_output=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
        timevariant_dt=True,
        timevariant_B=True,
        timevariant_C=True,
        use_D=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input if d_input is not None else d_model
        self.d_output = d_output if d_output is not None else d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_input, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = (
            nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if d_conv > 0
            else nn.Identity()
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.tv_dt, self.tv_B, self.tv_C = timevariant_dt, timevariant_B, timevariant_C
        self.tv_proj_dim = [0, 0, 0]
        if timevariant_dt:
            self.tv_proj_dim[0] = self.dt_rank
        if timevariant_B:
            self.tv_proj_dim[1] = self.d_state
        if timevariant_C:
            self.tv_proj_dim[2] = self.d_state
        self.x_proj = nn.Linear(self.d_inner, sum(self.tv_proj_dim), bias=False, **factory_kwargs) if sum(self.tv_proj_dim) > 0 else None

        if not timevariant_B:
            self.B = nn.Parameter(torch.rand(self.d_inner, self.d_state, **factory_kwargs))
            self.B._no_weight_decay = True
        if not timevariant_C:
            self.C = nn.Parameter(torch.rand(self.d_inner, self.d_state, **factory_kwargs))
            self.C._no_weight_decay = True

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        a_log = torch.log(a)
        self.A_log = nn.Parameter(a_log)
        self.A_log._no_weight_decay = True

        if use_D:
            self.D = nn.Parameter(torch.ones(self.d_inner, device=device)).float()
            self.D._no_weight_decay = True
        else:
            self.D = None

        self.out_proj = nn.Linear(self.d_inner, self.d_output, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        a = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        if (causal_conv1d_fn is None) or (self.d_conv not in [2, 3, 4]):
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        if self.x_proj is not None:
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, b_term, c_term = torch.split(x_dbl, self.tv_proj_dim, dim=-1)
        else:
            dt, b_term, c_term = None, None, None

        if not self.tv_dt:
            dt = torch.zeros(batch, self.d_inner, seqlen, device=self.dt_proj.bias.device, dtype=self.dt_proj.bias.dtype)
        else:
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        if not self.tv_B:
            b_term = self.B
        else:
            b_term = rearrange(b_term, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        if not self.tv_C:
            c_term = self.C
        else:
            c_term = rearrange(c_term, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            a,
            b_term,
            c_term,
            self.D,
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, b_term, c_term = torch.split(x_db, self.tv_proj_dim, dim=-1)

        if not self.tv_dt:
            dt = F.softplus(self.dt_proj.bias.to(dtype=dt.dtype))
        else:
            dt = F.linear(dt, self.dt_proj.weight)
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))

        if not self.tv_B:
            d_b = torch.einsum("bd,dn->bdn", dt, self.B)
        else:
            d_b = torch.einsum("bd,bn->bdn", dt, b_term)

        a = -torch.exp(self.A_log.float())
        d_a = torch.exp(torch.einsum("bd,dn->bdn", dt, a))
        ssm_state.copy_(ssm_state * d_a + rearrange(x, "b d -> b d 1") * d_b)

        if not self.tv_C:
            y = torch.einsum("bdn,dn->bd", ssm_state.to(dtype), self.C)
        else:
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), c_term)
        if self.D is not None:
            y = y + self.D.to(dtype) * x
        y = y * self.act(z)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype)
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
