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
We override v8DetectionLoss.__call__ to apply per-anchor area weights to
BOTH the classification BCE term and the bbox regression term, while
keeping target_scores in [0, 1] (required for BCE to remain mathematically
valid).

The earlier approach — multiplying target_scores by w in the assigner —
pushed targets to 4.0 and made BCE produce negative loss values, which
destroyed training. The correct approach is to leave target_scores as soft
labels in [0, 1] and multiply the per-anchor LOSS contributions instead.
"""

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

# Store original init_criterion so we can restore it
_original_init_criterion = None


def _area_weights(target_bboxes: torch.Tensor, a0: float, w_max: float) -> torch.Tensor:
    """Compute per-anchor area weight w = min(A0 / area, w_max).

    Computed in fp32 for numerical stability under AMP.
    Returns a tensor of shape [B, A] (same as fg_mask).
    """
    tb = target_bboxes.float()
    bw = (tb[..., 2] - tb[..., 0]).clamp(min=0)
    bh = (tb[..., 3] - tb[..., 1]).clamp(min=0)
    area = (bw * bh).clamp(min=1.0)
    return torch.clamp(a0 / area, max=w_max)


class AdaptiveDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with per-anchor area weights applied to BCE + bbox loss.

    Math
    ----
    Standard YOLO loss (per anchor):
        L_cls_i  = BCE(pred_score_i, target_score_i)
        L_bbox_i = (1 - IoU) * target_score_i + DFL_i

    Adaptive variant:
        w_i = min(A0 / area_i, w_max)       # only on foreground anchors
        L_cls_i  *= w_i
        L_bbox_i *= w_i
    """

    A0    = 32.0 * 32.0
    W_MAX = 4.0

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        # Use a per-element (un-reduced) BCE so we can apply per-anchor weights
        self._bce_none = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, preds, batch):  # noqa: C901
        # We delegate most of the heavy lifting to the parent by calling
        # the assigner ourselves, then computing both loss components with
        # per-anchor area weights. To stay version-robust, we reuse the
        # parent's helpers and field names.
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = self._make_anchors(feats)

        # Targets in image-pixel coords
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Predicted boxes in image-pixel coords
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Run the assigner (unmodified — gives us valid target_scores in [0, 1])
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ── Per-anchor area weights ───────────────────────────────────────────
        w = _area_weights(target_bboxes, self.A0, self.W_MAX)        # [B, A] fp32
        fg_bool = fg_mask.bool()
        w = torch.where(fg_bool, w, torch.ones_like(w))              # bg → 1.0
        w_dtype = w.to(dtype)                                        # match pred_scores

        # ── Classification loss (BCE, per-anchor weighted) ───────────────────
        # target_scores stays in [0, 1] — BCE math is correct.
        target_scores_sum = max(target_scores.sum(), 1)
        cls_per = self._bce_none(pred_scores, target_scores.to(dtype))   # [B, A, C]
        cls_loss = (cls_per * w_dtype.unsqueeze(-1)).sum() / target_scores_sum

        # ── Bbox + DFL loss (parent already weights by target_scores;
        #     we apply an extra per-anchor weight by scaling target_scores
        #     just for the foreground anchors before the bbox loss call) ─────
        target_bboxes_scaled = target_bboxes / stride_tensor
        if fg_mask.sum():
            # Scale target_scores → bbox loss gets weighted via target_scores.sum(-1)
            # This is safe because target_scores is only used as a weight inside
            # bbox_loss (not as a BCE target).
            weighted_target_scores = target_scores * w_dtype.unsqueeze(-1)
            weighted_target_scores_sum = max(weighted_target_scores.sum(), 1)

            box_loss, dfl_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_scaled,
                weighted_target_scores,
                weighted_target_scores_sum,
                fg_mask,
            )
        else:
            box_loss = pred_scores.new_zeros(1)
            dfl_loss = pred_scores.new_zeros(1)

        loss = pred_scores.new_zeros(3)
        loss[0] = box_loss * self.hyp.box
        loss[1] = cls_loss * self.hyp.cls
        loss[2] = dfl_loss * self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    # ── Helpers that exist in v8DetectionLoss but with different names
    #    across Ultralytics versions ────────────────────────────────────────
    def _make_anchors(self, feats):
        """Build anchor_points / stride_tensor — falls back to ultralytics helper."""
        from ultralytics.utils.tal import make_anchors
        return make_anchors(feats, self.stride, 0.5)


def enable_adaptive_loss(a0: float = 32 * 32, w_max: float = 4.0):
    """Monkey-patch DetectionModel.init_criterion to use AdaptiveDetectionLoss.

    Must be called BEFORE the YOLO model is loaded for training.
    Call disable_adaptive_loss() to restore original behavior.
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

    Must be called before training a baseline model when adaptive was previously
    enabled in the same Python process (e.g., in --all sequential training).
    """
    global _original_init_criterion
    if _original_init_criterion is None:
        return

    from ultralytics.nn.tasks import DetectionModel
    DetectionModel.init_criterion = _original_init_criterion
    print("[adaptive_loss] DISABLED: restored standard loss")


# ── Legacy class (kept for backward compat with existing imports) ────────────
class AdaptiveAssigner(TaskAlignedAssigner):
    """Deprecated: no longer used. Kept so old imports do not break."""
    pass
