from __future__ import annotations

import functools
import inspect
from pathlib import Path

import torch
from asteroid_filterbanks import make_enc_dec
from torch import nn


def is_tracing() -> bool:
    return torch._C._is_tracing()


def script_if_tracing(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            return fn(*args, **kwargs)
        compiled_fn = torch.jit.script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn
    return wrapper


@script_if_tracing
def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


@script_if_tracing
def jitable_shape(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(tensor.shape)


@script_if_tracing
def _unsqueeze_to_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    return x


class _LayerNorm(nn.Module):
    def __init__(self, channel_size: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x: torch.Tensor) -> torch.Tensor:
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        dims = torch.arange(1, len(x.shape)).tolist()
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / torch.sqrt(var + eps))


class ChanLN(_LayerNorm):
    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / torch.sqrt(var + eps))


class CumLN(_LayerNorm):
    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(
            start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / torch.sqrt(cum_var + eps))


def get_norm(identifier: str | None):
    if identifier is None:
        return None
    registry = {"gLN": GlobLN, "cLN": ChanLN, "cgLN": CumLN}
    if callable(identifier):
        return identifier
    if identifier not in registry:
        raise ValueError(f"Could not interpret normalization identifier: {identifier}")
    return registry[identifier]


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def get_activation(identifier: str):
    registry = {
        "linear": nn.Identity,
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": Swish,
        "softmax": nn.Softmax,
    }
    if callable(identifier):
        return identifier
    if identifier not in registry:
        raise ValueError(f"Could not interpret activation identifier: {identifier}")
    return registry[identifier]


def has_arg(fn, name: str) -> bool:
    if inspect.isclass(fn):
        fn = fn.__init__
    return name in inspect.signature(fn).parameters


class _Chop1d(nn.Module):
    def __init__(self, chop_size: int):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_out_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: str = "gLN",
        causal: bool = False,
    ):
        super().__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = get_norm(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x: torch.Tensor):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_src: int,
        out_chan: int | None = None,
        n_blocks: int = 8,
        n_repeats: int = 3,
        bn_chan: int = 128,
        hid_chan: int = 512,
        skip_chan: int = 128,
        conv_kernel_size: int = 3,
        norm_type: str = "gLN",
        mask_act: str = "relu",
        causal: bool = False,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan if out_chan else in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = get_norm(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for _ in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2**x
                if not causal:
                    padding //= 2
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * self.out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = get_activation(mask_act)
        self.output_act = mask_nl_class(dim=1) if has_arg(mask_nl_class, "dim") else mask_nl_class()

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        return self.output_act(score)


@script_if_tracing
def _shape_reconstructed(reconstructed: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


class BaseEncoderMaskerDecoder(nn.Module):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = get_activation(encoder_activation or "linear")()

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        return self.enc_activation(self.encoder(wav))

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        return self.decoder(masked_tf_rep)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)


class ConvTasNet(BaseEncoderMaskerDecoder):
    def __init__(
        self,
        n_src: int,
        out_chan: int | None = None,
        n_blocks: int = 8,
        n_repeats: int = 3,
        bn_chan: int = 128,
        hid_chan: int = 512,
        skip_chan: int = 128,
        conv_kernel_size: int = 3,
        norm_type: str = "gLN",
        mask_act: str = "sigmoid",
        in_chan: int | None = None,
        causal: bool = False,
        fb_name: str = "free",
        kernel_size: int = 16,
        n_filters: int = 512,
        stride: int = 8,
        encoder_activation=None,
        sample_rate: float = 8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None and in_chan != n_feats:
            raise AssertionError(
                f"Filterbank output channels and input channels should match: {n_feats} vs {in_chan}"
            )
        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
        masker = TDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)

    @classmethod
    def from_pretrained_package(cls, package: dict) -> "ConvTasNet":
        model = cls(**package["model_args"])
        model.load_state_dict(package["state_dict"])
        return model

    @classmethod
    def from_pretrained_path(cls, package_path: str | Path) -> "ConvTasNet":
        package = torch.load(package_path, map_location="cpu", weights_only=False)
        return cls.from_pretrained_package(package)
