import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: pip install pytorch-msssim
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
    Composite reconstruction loss for 2-channel 3D microscopy volumes in [0,1].
    Accepts (B, T, 2, D, H, W). Flattens time for MS-SSIM.

    L = w_ssim * (1 - MS_SSIM) + w_l1 * MAE + w_mse * MSE + w_grad * GradLoss + w_tv * TV (optional)

    Defaults emphasize structural fidelity and edge preservation for high SNR data.
    """
    def __init__(
        self,
        w_ssim: float = 0.5,
        w_l1: float   = 0.25,
        w_mse: float  = 0.25,
        w_grad: float = 0.0,
        w_tv: float   = 0.0,        # enable only if you observe speckle/checkerboard
        use_mask: bool = False,      # focus loss where signal exists
        mask_thresh: float = 0.05,  # soft threshold on target intensity
        mask_softness: float = 40.0,
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
            # MS-SSIM over 3D with 2 channels; expects (N, C, D, H, W)
            self.ms_ssim = MS_SSIM(
                data_range=data_range,
                size_average=True,
                win_size=11,
                channel=2,
                spatial_dims=3
            )
        else:
            self.ms_ssim = None

    @staticmethod
    def _soft_mask(x: torch.Tensor, thresh: float, softness: float):
        # x in [0,1]; emphasize foreground (> thresh) with a soft mask
        return torch.sigmoid(softness * (x - thresh))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8):
        if mask is None:
            return x.mean()
        return (x * mask).sum() / (mask.sum() + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, T, 2, D, H, W) in [0,1]
        """
        assert pred.shape == target.shape, "pred/target shape mismatch"
        assert pred.ndim == 6 and pred.size(2) == 2, "Expected (B,T,2,D,H,W)"

        B, T, C, D, H, W = pred.shape
        device = pred.device

        # Foreground soft mask (computed on target and broadcast)
        mask_btcdhw = None
        if self.use_mask:
            # (B,T,2,D,H,W) -> (B,T,2,D,H,W)
            mask_btcdhw = self._soft_mask(target, self.mask_thresh, self.mask_softness).detach()

        # --- MS-SSIM over 3D per (B*T) volume ---
        if self.ms_ssim is not None and self.w_ssim > 0:
            pred_bt = pred.reshape(B * T, C, D, H, W)
            targ_bt = target.reshape(B * T, C, D, H, W)
            ssim_val = self.ms_ssim(pred_bt, targ_bt)  # scalar in [0,1]
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = torch.tensor(0.0, device=device, dtype=pred.dtype)

        # --- Pixel-wise losses ---
        diff = pred - target
        l1_map = diff.abs()
        l2_map = diff.square()

        loss_l1  = self._masked_mean(l1_map, mask_btcdhw)
        loss_mse = self._masked_mean(l2_map, mask_btcdhw)

        # --- Gradient consistency (edge/detail preservation) ---
        # Collapse time into batch for gradient computation
        pred_bt = pred.reshape(B * T, C, D, H, W)
        targ_bt = target.reshape(B * T, C, D, H, W)

        if self.w_grad > 0:
            pD, pH, pW = _finite_diff_3d(pred_bt)
            tD, tH, tW = _finite_diff_3d(targ_bt)
            gdiff = (pD - tD).abs() + (pH - tH).abs() + (pW - tW).abs()  # (B*T,C,D,H,W)

            if mask_btcdhw is not None:
                mask_btchw = mask_btcdhw.reshape(B * T, C, D, H, W)
            else:
                mask_btchw = None

            loss_grad = self._masked_mean(gdiff, mask_btchw)
        else:
            loss_grad = torch.tensor(0.0, device=device, dtype=pred.dtype)

        # --- Optional TV regularizer (very light if enabled) ---
        if self.w_tv > 0:
            tv = pD.abs().mean() + pH.abs().mean() + pW.abs().mean()
        else:
            tv = torch.tensor(0.0, device=device, dtype=pred.dtype)

        # --- Final weighted sum ---
        loss = (
            self.w_ssim * loss_ssim +
            self.w_l1   * loss_l1   +
            self.w_mse  * loss_mse  +
            self.w_grad * loss_grad +
            self.w_tv   * tv
        )

        # Metrics dict for logging
        metrics = {
            "recon_loss": loss.detach(),
            "ms_ssim_term": loss_ssim.detach(),
            "l1_term": loss_l1.detach(),
            "mse_term": loss_mse.detach(),
            "grad_term": loss_grad.detach(),
            "tv_term": tv.detach(),
        }
        return loss, metrics


# ----------------------- Example usage -----------------------
# if __name__ == "__main__":
#     # Dummy example matching your model I/O
#     B, T, C, D, H, W = 1, 10, 2, 30, 128, 128  # example sizes
#     x = torch.rand(B, T, C, D, H, W, device="cuda")  # in [0,1]

#     from torchsummary import summary
#     class MitoSpace3DEncoder(nn.Module): ...
#     class MitoSpace3DDecoder(nn.Module): ...
#     class MitoSpace3DAutoencoder(nn.Module): ...

#     # Assuming your classes are defined/imported above, construct and forward:
#     # auto = MitoSpace3DAutoencoder().cuda()
#     # y = auto(x)

#     # For demo without the model, pretend y ~ x:
#     y = x.clone()

#     criterion = ReconstructionLoss3DTime(
#         w_ssim=0.5, w_l1=0.25, w_mse=0.15, w_grad=0.10, w_tv=0.0,
#         use_mask=True, mask_thresh=0.05, mask_softness=40.0, data_range=1.0
#     ).cuda()

#     loss, metrics = criterion(y, x)
#     print("loss:", float(loss))
#     print({k: float(v) for k, v in metrics.items()})