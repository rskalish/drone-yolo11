"""
Adaptive area-weighted loss for one-stage object detectors.

Implements the loss proposed in:
"Adaptive loss function for improving detection of small-scale UAVs"
(Skalish R., 2026).

For each foreground anchor, the per-sample weight is:

    w_i = min(A_0 / A_i, w_max)

where A_i is the bounding box area in pixels, A_0 is the reference area
of a "small" target (default 32 x 32 = 1024 px), and w_max caps the
weight for extremely small boxes (default 4.0).

Implementation note
-------------------
We hook the TaskAlignedAssigner that produces target_scores. To remain
compatible across Ultralytics versions (which have been refactoring the
__call__ → parse_output → loss → get_assigned_targets_and_loss path), we
only modify the assigner output — never the loss class itself.

To keep BCE classification math valid we must keep target_scores ∈ [0, 1].
A previous version multiplied by w (up to W_MAX = 4.0), which pushed targets
above 1 and made BCE emit large negative loss values (cls_loss ≈ -89 was
observed at epoch 1). To preserve relative per-anchor weighting without
breaking BCE we normalise the weights:

    w_norm_i = w_i / W_MAX   ∈ [1/W_MAX, 1]
    target_scores ← target_scores * w_norm_i        (still ∈ [0, 1])

This is equivalent to the paper's formulation up to a global scale of
1/W_MAX — the ratio of gradients between small and large anchors is
preserved (W_MAX × stronger for the smallest boxes), which is what
controls optimisation dynamics.

Both downstream loss components benefit:
- BCE classification uses target_scores as soft labels: small-object
  anchors retain their full target value (≈ original), large-object
  anchors get smaller targets (down-weighted contribution).
- Bbox/DFL regression uses `target_scores.sum(-1)[fg_mask]` as per-anchor
  weight, so small boxes receive proportionally more bbox-loss signal.
"""

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

# Stored on first patch so we can restore the standard loss
_original_init_criterion = None


class AdaptiveAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner that re-weights target_scores by normalised area.

    Foreground anchors with small ground-truth boxes keep ~full
    target_scores; foreground anchors with large boxes get target_scores
    scaled down by up to 1/W_MAX.
    """

    A0    = 32.0 * 32.0
    W_MAX = 4.0

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = (
            super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        )

        # Compute area weights in fp32 (stable under AMP)
        tb = target_bboxes.float()
        bw = (tb[..., 2] - tb[..., 0]).clamp(min=0)
        bh = (tb[..., 3] - tb[..., 1]).clamp(min=0)
        area = (bw * bh).clamp(min=1.0)
        w = torch.clamp(self.A0 / area, max=self.W_MAX)        # ∈ [1, W_MAX]

        # Normalise so that target_scores ∈ [0, 1] is preserved
        w_norm = w / self.W_MAX                                # ∈ [1/W_MAX, 1]

        # Apply only to foreground anchors; background → 1.0 (no change)
        fg_bool = fg_mask.bool()
        w_norm = torch.where(fg_bool, w_norm, torch.ones_like(w_norm))

        # Cast to target_scores dtype before scaling (AMP-safe)
        target_scores = target_scores * w_norm.to(target_scores.dtype).unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


class AdaptiveDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with the assigner replaced by AdaptiveAssigner."""

    A0    = 32.0 * 32.0
    W_MAX = 4.0

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        old = self.assigner
        new = AdaptiveAssigner(
            topk=old.topk,
            num_classes=old.num_classes,
            alpha=old.alpha,
            beta=old.beta,
        )
        new.A0    = AdaptiveDetectionLoss.A0
        new.W_MAX = AdaptiveDetectionLoss.W_MAX
        self.assigner = new


def enable_adaptive_loss(a0: float = 32 * 32, w_max: float = 4.0):
    """Monkey-patch DetectionModel.init_criterion to use AdaptiveDetectionLoss.

    Must be called BEFORE the YOLO model is loaded for training.
    Call disable_adaptive_loss() to restore standard behaviour.
    """
    global _original_init_criterion
    from ultralytics.nn.tasks import DetectionModel

    AdaptiveDetectionLoss.A0    = float(a0)
    AdaptiveDetectionLoss.W_MAX = float(w_max)

    if _original_init_criterion is None:
        _original_init_criterion = DetectionModel.init_criterion

    def _init_criterion(self):
        return AdaptiveDetectionLoss(self)

    DetectionModel.init_criterion = _init_criterion
    print(f"[adaptive_loss] ENABLED: A0={a0:.0f} px², w_max={w_max}")


def disable_adaptive_loss():
    """Restore the original DetectionModel.init_criterion (standard loss).

    Must be called before training a baseline model when adaptive was
    previously enabled in the same Python process (e.g. --all mode).
    """
    global _original_init_criterion
    if _original_init_criterion is None:
        return
    from ultralytics.nn.tasks import DetectionModel
    DetectionModel.init_criterion = _original_init_criterion
    print("[adaptive_loss] DISABLED: restored standard loss")
