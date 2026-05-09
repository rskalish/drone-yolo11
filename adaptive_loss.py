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
Instead of copying the entire v8DetectionLoss.__call__ (which can break
across Ultralytics versions), we hook the TaskAlignedAssigner that
produces target_scores. We multiply target_scores by the adaptive
weight, and that propagates through both the bbox-regression loss and
the classification loss naturally, since both are normalised via
target_scores_sum.
"""

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class AdaptiveAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner that scales target_scores by min(A0/A_i, w_max)."""

    A0 = 32.0 * 32.0
    W_MAX = 4.0

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = (
            super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        )

        # target_bboxes here are in pixel coordinates at imgsz scale (xyxy)
        bw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0)
        bh = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0)
        area = (bw * bh).clamp(min=1.0)
        w = torch.clamp(self.A0 / area, max=self.W_MAX)
        w = torch.where(fg_mask, w, torch.ones_like(w))
        target_scores = target_scores * w.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


class AdaptiveDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with the assigner replaced by AdaptiveAssigner."""

    A0 = 32.0 * 32.0
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
        new.A0 = AdaptiveDetectionLoss.A0
        new.W_MAX = AdaptiveDetectionLoss.W_MAX
        self.assigner = new


def enable_adaptive_loss(a0: float = 32 * 32, w_max: float = 4.0):
    """Monkey-patch DetectionModel.init_criterion to use AdaptiveDetectionLoss.

    Must be called BEFORE the YOLO model is loaded for training.
    """
    from ultralytics.nn.tasks import DetectionModel

    AdaptiveDetectionLoss.A0 = float(a0)
    AdaptiveDetectionLoss.W_MAX = float(w_max)

    def _init_criterion(self):
        return AdaptiveDetectionLoss(self)

    DetectionModel.init_criterion = _init_criterion
    print(f"[adaptive_loss] enabled: A0={a0:.0f}, w_max={w_max}")
