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

The weight is multiplied into target_scores, so it propagates through
both the bbox regression loss and the classification loss, matching the
formulation L_adapt = sum_i w_i * (lambda_loc*L_loc + lambda_obj*L_obj +
lambda_cls*L_cls).
"""

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class AdaptiveDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with adaptive per-sample weighting based on box area."""

    A0 = 32.0 * 32.0
    W_MAX = 4.0

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ---------------- adaptive area-based reweighting ----------------
        bw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0)
        bh = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0)
        area = (bw * bh).clamp(min=1.0)
        w = torch.clamp(self.A0 / area, max=self.W_MAX)
        w = torch.where(fg_mask, w, torch.ones_like(w))
        target_scores = target_scores * w.unsqueeze(-1)
        # -----------------------------------------------------------------

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


def enable_adaptive_loss(a0: float = 32 * 32, w_max: float = 4.0):
    """Monkey-patch DetectionModel.init_criterion to use AdaptiveDetectionLoss.

    Must be called BEFORE YOLO model is loaded for training.
    """
    from ultralytics.nn.tasks import DetectionModel

    AdaptiveDetectionLoss.A0 = float(a0)
    AdaptiveDetectionLoss.W_MAX = float(w_max)

    def _init_criterion(self):
        return AdaptiveDetectionLoss(self)

    DetectionModel.init_criterion = _init_criterion
    print(f"[adaptive_loss] enabled: A0={a0:.0f}, w_max={w_max}")
