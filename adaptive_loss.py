"""
Adaptive area-weighted loss for one-stage object detectors.

Implements the loss proposed in:
"Adaptive loss function for improving detection of small-scale UAVs"
(Skalish R., 2026).

For each foreground anchor, the per-sample weight is:

    w_i = min(A_0 / A_i, w_max)

where A_i is the bounding box area in pixels, A_0 is the reference area
of a "small" target (default 32 × 32 = 1024 px), and w_max caps the
weight for extremely small boxes (default 4.0).

Implementation: "boost-only" target_score adjustment
-----------------------------------------------------
We hook the TaskAlignedAssigner that produces target_scores and apply
the area weights as a "boost" that PULLS target_scores toward 1.0 for
small objects, but leaves target_scores UNCHANGED for large objects:

    boost_i = (w_i - 1) / (w_max - 1)               # 0 for large, 1 for very small
    target_scores_i ← target_scores_i + (1 - target_scores_i) · boost_i

This way:
- Large objects (w_i = 1):   boost = 0 → target_scores unchanged.
  Large-object detection is fully preserved.
- Small objects (w_i = w_max): boost = 1 → target_scores → 1.0.
  Strong, max-confidence training signal.
- Medium: partial boost proportional to size deficit.

The previous normalisation (target_scores ×= w / w_max) was rejected
because it REDUCED target_scores for large objects, training the model
to produce low confidence for them and crashing large-object recall.
The boost formulation never decreases any target — it only adds.

Both downstream loss components benefit:
- BCE classification uses target_scores as soft labels: small-object
  anchors get pulled toward 1.0 → stronger gradient.
- Bbox/DFL regression uses target_scores.sum(-1)[fg_mask] as per-anchor
  weight, so small boxes receive proportionally more bbox-loss signal,
  large boxes keep their original weight.
"""

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

# Stored on first patch so we can restore the standard loss
_original_init_criterion = None


class AdaptiveAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner that boosts target_scores for small-object anchors.

    A `boost_strength` ∈ [0, 1] controls how strongly small-object targets
    are pulled toward 1.0. With strength=0.3 (default), a very-small anchor's
    target_score is pulled 30% of the way to 1.0 — strong enough to improve
    small-object recall, gentle enough to avoid over-confidence (which would
    inflate false positives and drop Precision).
    """

    A0              = 32.0 * 32.0
    W_MAX           = 4.0
    BOOST_STRENGTH  = 0.3      # how aggressively to pull target_scores toward 1.0

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = (
            super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        )

        # Area weights in fp32 (stable under AMP)
        tb = target_bboxes.float()
        bw = (tb[..., 2] - tb[..., 0]).clamp(min=0)
        bh = (tb[..., 3] - tb[..., 1]).clamp(min=0)
        area = (bw * bh).clamp(min=1.0)
        w = torch.clamp(self.A0 / area, max=self.W_MAX)            # ∈ [1, W_MAX]

        # Boost factor: 0 for large (w=1), 1 for very small (w=W_MAX)
        denom = max(self.W_MAX - 1.0, 1e-6)
        boost = ((w - 1.0) / denom).clamp(0.0, 1.0) * self.BOOST_STRENGTH

        # Apply only to foreground anchors; background → boost=0 (no change)
        fg_bool = fg_mask.bool()
        boost = torch.where(fg_bool, boost, torch.zeros_like(boost))

        # Pull target_scores toward 1.0 by `boost`; never decrease them
        boost = boost.to(target_scores.dtype).unsqueeze(-1)
        target_scores = target_scores + (1.0 - target_scores) * boost

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
    print(f"[adaptive_loss] ENABLED: A0={a0:.0f} px², w_max={w_max} (boost-only)")


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
