import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import MS_SSIM
    _HAS_MSSSIM = True
except Exception:
    _HAS_MSSSIM = False


def _finite_diff_3d(x: torch.Tensor):
    """
    Forward differences along (D, H, W) with padding to original size.
    x: (N, C, D, H, W)
    returns: (gD, gH, gW) each (N, C, D, H, W)
    """
    gD = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    gH = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    gW = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    gD = F.pad(gD, (0, 0, 0, 0, 0, 1))
    gH = F.pad(gH, (0, 0, 0, 1, 0, 0))
    gW = F.pad(gW, (0, 1, 0, 0, 0, 0))
    return gD, gH, gW


class ReconstructionLoss(nn.Module):
    """
    Composite reconstruction loss for 3D volumes in [0,1].
    Accepts (B, C, D, H, W).

    L = w_ssim * (1 - MS_SSIM) + w_l1 * MAE + w_mse * MSE + w_grad * GradLoss + w_tv * TV (optional)

    Defaults emphasize structural fidelity and edge preservation for high SNR data.
    """
    def __init__(
        self,
        n_channels: int = 1,
        w_ssim: float = 0.5,
        w_l1: float   = 0.25,
        w_mse: float  = 0.25,
        w_grad: float = 0.0,
        w_tv: float   = 0.0,
        use_mask: bool = False,
        mask_thresh: float = 0.05,
        mask_softness: float = 40.0,
        win_size_ssim: int = 7,
        data_range: float = 1.0
    ):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1   = w_l1
        self.w_mse  = w_mse
        self.w_grad = w_grad
        self.w_tv   = w_tv
        self.use_mask = use_mask
        self.mask_thresh = mask_thresh
        self.mask_softness = mask_softness
        self.data_range = data_range

        if _HAS_MSSSIM and self.w_ssim > 0:
            self.ms_ssim = MS_SSIM(
                data_range=data_range,
                size_average=True,
                win_size=win_size_ssim,
                channel=n_channels,
                spatial_dims=3
            )
        else:
            self.ms_ssim = None


    @staticmethod
    def _soft_mask(x: torch.Tensor, thresh: float, softness: float):
        return torch.sigmoid(softness * (x - thresh))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8):
        if mask is None:
            return x.mean()
        return (x * mask).sum() / (mask.sum() + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, C, D, H, W) in [0,1]
        """
        assert pred.shape == target.shape, "pred/target shape mismatch"
        assert pred.ndim == 5, "Expected (B,C,D,H,W)"

        B, C, D, H, W = pred.shape
        device = pred.device

        mask_bcdhw = None
        if self.use_mask:
            mask_bcdhw = self._soft_mask(target, self.mask_thresh, self.mask_softness).detach()

        if self.ms_ssim is not None and self.w_ssim > 0:
            ssim_val = self.ms_ssim(pred, target)
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = torch.tensor(0.0, device=device, dtype=pred.dtype)

        diff = pred - target
        l1_map = diff.abs()
        l2_map = diff.square()

        loss_l1  = self._masked_mean(l1_map, mask_bcdhw)
        loss_mse = self._masked_mean(l2_map, mask_bcdhw)

        if self.w_grad > 0:
            pD, pH, pW = _finite_diff_3d(pred)
            tD, tH, tW = _finite_diff_3d(target)
            gdiff = (pD - tD).abs() + (pH - tH).abs() + (pW - tW).abs()  # (B,C,D,H,W)

            loss_grad = self._masked_mean(gdiff, mask_bcdhw)
        else:
            loss_grad = torch.tensor(0.0, device=device, dtype=pred.dtype)

        if self.w_tv > 0:
            tv = pD.abs().mean() + pH.abs().mean() + pW.abs().mean()
        else:
            tv = torch.tensor(0.0, device=device, dtype=pred.dtype)

        loss = (
            self.w_ssim * loss_ssim +
            self.w_l1   * loss_l1   +
            self.w_mse  * loss_mse  +
            self.w_grad * loss_grad +
            self.w_tv   * tv
        )

        metrics = {
            "recon_loss": loss.detach(),
            "ms_ssim_term": loss_ssim.detach(),
            "l1_term": loss_l1.detach(),
            "mse_term": loss_mse.detach(),
            "grad_term": loss_grad.detach(),
            "tv_term": tv.detach(),
        }
        return loss, metrics