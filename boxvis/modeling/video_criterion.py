# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample
)

from mask2former_video.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_with_weight_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    weights = weights.flatten(1)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    return loss.sum() / max(num_masks, 0.5)


sigmoid_ce_with_weight_loss_jit = torch.jit.script(
    sigmoid_ce_with_weight_loss
)  # type: torch.jit.ScriptModule


def dice_coefficient_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. Nx...
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs**2).sum(-1) + (targets**2).sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-6)
    return loss.sum() / num_masks


dice_coefficient_loss_jit = torch.jit.script(
    dice_coefficient_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class BoxVISTeacherSetPseudoMask(nn.Module):
    def __init__(self, matcher):
        """Create the criterion.
        Parameters:
            matcher: matching objects between targets and proposals
        """
        super().__init__()
        self.matcher = matcher

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc.
                      The mask in targets is generated mask via bounding boxes
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        targets = self.set_pseudo_mask(indices, outputs_without_aux, targets)
        for bs, target in enumerate(targets):
            tgt_mask_box = target["masks"]
            tgt_mask_pseudo = target["masks_pseudo"]
            tgt_mask_pseudo_score = target["mask_pseudo_scores"]

            tgt_h, tgt_w = tgt_mask_box.shape[-2:]
            tgt_mask_pseudo = F.interpolate(tgt_mask_pseudo, (tgt_h, tgt_w),
                                            mode='bilinear', align_corners=False)  # cQ, T, Ht, Wt

            #  project term --------------------------
            tgt_mask_pseudo_y = tgt_mask_pseudo.sigmoid().max(dim=-2, keepdim=True)[0].flatten(1)
            tgt_mask_box_y = tgt_mask_box.max(dim=-2, keepdim=True)[0].flatten(1)
            numerator = 2 * (tgt_mask_pseudo_y * tgt_mask_box_y).sum(-1)
            denominator = (tgt_mask_pseudo_y ** 2).sum(-1) + (tgt_mask_box_y ** 2).sum(-1)
            mask_proj_y = numerator / denominator.clamp(min=1e-6)

            tgt_mask_pseudo_x = tgt_mask_pseudo.sigmoid().max(dim=-1, keepdim=True)[0].flatten(1)
            tgt_mask_box_x = tgt_mask_box.max(dim=-1, keepdim=True)[0].flatten(1)
            numerator = 2 * (tgt_mask_pseudo_x * tgt_mask_box_x).sum(-1)
            denominator = (tgt_mask_pseudo_x ** 2).sum(-1) + (tgt_mask_box_x ** 2).sum(-1)
            mask_proj_x = numerator / denominator.clamp(min=1e-6)

            mask_proj_score = 0.5 * (mask_proj_x + mask_proj_y)

            target["mask_pseudo_scores"] = tgt_mask_pseudo_score * mask_proj_score
            target["masks_pseudo"] = tgt_mask_box * tgt_mask_pseudo.sigmoid()

        return targets

    def set_pseudo_mask(self, indices, outputs, targets):
        src_masks = outputs["pred_masks"].clone().detach()  # B, cQ, T, Hp, Wp
        src_logits = outputs["pred_logits"].softmax(dim=-1).clone().detach()  # B, cQ, k

        for i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
            assert len(tgt_idx) == target["masks"].shape[0]
            tgt_idx, tgt_idx_sorted = tgt_idx.sort()
            src_idx = src_idx[tgt_idx_sorted]
            tgt_labels = target["labels"]
            target["mask_pseudo_scores"] = src_logits[i, src_idx, tgt_labels]
            target["masks_pseudo"] = src_masks[i, src_idx]

        return targets


class BoxVISVideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_frames,
                 num_points, oversample_ratio, importance_sample_ratio, is_coco=True,
                 boxvis_enabled=False,  boxvis_pairwise_enable=True, boxvis_pairwise_num_stpair=7,
                 boxvis_pairwise_dilation=4, boxvis_pairwise_color_thresh=0.3,
                 boxvis_pairwise_corr_kernel_size=3, boxvis_pairwise_corr_stride=2, boxvis_pairwise_corr_thresh=0.9,
                 boxvis_ema_enabled=False, boxvis_pseudo_mask_score_thresh=0.5, max_iters=10000,
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses
            boxvis_enabled: It controls the annotation types: pixel-wise or box-level annotations for VIS task
            boxvis_ema_enabled: It controls whether to use Teacher Net to produce pseudo instance masks for VIS task
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_frames = num_frames
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.is_coco = is_coco

        # box-supervised video instance segmentation
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_pairwise_enable = boxvis_pairwise_enable
        self.boxvis_pairwise_dilation = boxvis_pairwise_dilation
        self.boxvis_pairwise_color_thresh = boxvis_pairwise_color_thresh
        self.boxvis_pairwise_corr_kernel_size = boxvis_pairwise_corr_kernel_size
        self.boxvis_pairwise_corr_stride = boxvis_pairwise_corr_stride
        self.boxvis_pairwise_corr_thresh = boxvis_pairwise_corr_thresh
        self.register_buffer("_iter", torch.zeros([1]))

        # Teacher net to produce pseudo masks
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.pseudo_mask_score_thresh = boxvis_pseudo_mask_score_thresh
        self.max_iters = max_iters

        # spatial-temporal pairwise loss
        self.unfold_patch = nn.Unfold(kernel_size=(boxvis_pairwise_corr_kernel_size, boxvis_pairwise_corr_kernel_size),
                                      padding=boxvis_pairwise_corr_kernel_size//2, stride=boxvis_pairwise_corr_stride)
        self.unfold_pixel = nn.Unfold(kernel_size=(1, 1), padding=0, stride=boxvis_pairwise_corr_stride)
        d = max(boxvis_pairwise_dilation // boxvis_pairwise_corr_stride, 1)
        if boxvis_pairwise_num_stpair == 2:
            self.enable_patch_corr = False
            self.enable_box_center_shifting = False
            self.pixel_offsets = torch.as_tensor([[0, d, 0], [0, 0, d]]).reshape(-1, 3)
        elif boxvis_pairwise_num_stpair == 4:
            self.enable_patch_corr = False
            self.enable_box_center_shifting = False
            self.pixel_offsets = torch.as_tensor([[0, d, 0], [0, 0, d], [0, -d, d], [0, d, d]]).reshape(-1, 3)
        elif boxvis_pairwise_num_stpair == 5:
            self.enable_patch_corr = True
            self.enable_box_center_shifting = True
            self.pixel_offsets = torch.as_tensor([[0, d, 0], [0, 0, d],
                                                  [1, 0, 0], [1, d, 0], [1, 0, d]]).reshape(-1, 3)
        elif boxvis_pairwise_num_stpair == 7:
            self.enable_patch_corr = True
            self.enable_box_center_shifting = True
            self.pixel_offsets = torch.as_tensor([[0, d, 0], [0, 0, d],
                                                  [1, 0, 0], [1, d, 0], [1, 0, d], [1, -d, 0], [1, 0, -d]]
                                                 ).reshape(-1, 3)

        elif boxvis_pairwise_num_stpair == 9:
            self.enable_patch_corr = True
            self.enable_box_center_shifting = True
            self.pixel_offsets = torch.as_tensor([[0, d, 0], [0, 0, d], [0, -d, d], [0, d, d],
                                                  [1, 0, 0], [1, d, 0], [1, 0, d], [1, -d, 0], [1, 0, -d]]
                                                 ).reshape(-1, 3)
        else:
            raise ValueError

    def loss_labels(self, outputs, targets, indices, num_masks, l_layer):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # BxQxK

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords: NT x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_with_box_supervised(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the masks with only box annotations: the projection loss and the pairwise loss.
        If enabling Teacher Net with EMA, the pseudo mask supervision includes the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]  # NxTxHpxWp
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxTxHxW
        tgt_h, tgt_w = target_masks.shape[-2:]

        # the predicted masks are inaccurate in the initial iterations
        if self.is_coco or max(tgt_h, tgt_w) > 480:
            h_, w_ = int(tgt_h/2), int(tgt_w/2)
            with torch.no_grad():
                target_masks = F.interpolate(target_masks, (h_, w_),
                                             mode='bilinear', align_corners=False)
                src_masks = F.interpolate(src_masks, (h_, w_),
                                          mode='bilinear', align_corners=False)
        else:
            # upsampling for videos in VIS videos with low resolution input
            src_masks = F.interpolate(src_masks, (tgt_h, tgt_w), mode='bilinear', align_corners=False)

        # ----------------------- points out of bounding box / project term --------------------------
        # max simply selects the greatest value to back-prop, so max is the identity operation for that one element
        mask_losses_y = dice_coefficient_loss_jit(
            src_masks.sigmoid().max(dim=-2, keepdim=True)[0].flatten(1),
            target_masks.max(dim=-2, keepdim=True)[0].flatten(1),
            num_masks
        )
        mask_losses_x = dice_coefficient_loss_jit(
            src_masks.sigmoid().max(dim=-1, keepdim=True)[0].flatten(1),
            target_masks.max(dim=-1, keepdim=True)[0].flatten(1),
            num_masks
        )
        loss_proj = mask_losses_x + mask_losses_y
        losses = {'loss_mask_proj': loss_proj}

        if self.boxvis_pairwise_enable:
            losses['loss_mask_pair'] = self.spatial_temporal_pairwise_loss(outputs, targets, indices)

        if self.boxvis_ema_enabled:
            losses.update(self.loss_masks_pseudo(outputs, targets, indices, num_masks))

        return losses

    def spatial_temporal_pairwise_loss(self, outputs, targets, indices):
        """Compute the pairwise mask loss without pixel-wise annotation: spatial-temporal pairwise loss.
        """
        pred_masks = outputs["pred_masks"]  # bsxQxTxHpxWp

        losses = []
        for (src_idx, tgt_idx), src_masks, t in zip(indices, pred_masks, targets):
            if t['masks'].nelement() == 0:
                continue

            H, W = t['st_pair_spatial_size']
            dH, dW = t['st_pair_unfold_size']
            indices_ctr_shift = t['st_pair_indices_ctr_shift'].flatten(1)  # NxTHW
            is_inbox_pairwise_st = t['st_pair_is_inbox_pairwise']  # n_o [NxTHW]

            src_masks = src_masks[src_idx]  # NxTxHxW
            indices_ctr_shift = indices_ctr_shift[tgt_idx]  # NxTHW
            is_inbox_pairwise_st = [x[tgt_idx].flatten() for x in is_inbox_pairwise_st]

            # upsampling for videos in VIS videos with low resolution input
            src_masks = F.interpolate(src_masks, (H, W), mode='bilinear', align_corners=False)
            # the mean mask of all pixels in its centered patch
            src_masks = rearrange(self.unfold_pixel(src_masks),
                                  'N (T K) (H W) -> N T H W K',
                                  H=dH, W=dW, K=1).mean(dim=-1)  # NxTxHpxWp

            # box-center guided shifting between the t and t+1 frames
            src_masks_ctr_shift = torch.zeros_like(src_masks.flatten(1))  # NxTHW
            for obj_i, indices_ctr_shift_i in enumerate(indices_ctr_shift):
                keep_next = indices_ctr_shift_i >= 0
                indices_next = indices_ctr_shift_i[keep_next]
                src_masks_ctr_shift[obj_i, keep_next] = src_masks[obj_i].flatten()[indices_next]
            src_masks_ctr_shift = rearrange(src_masks_ctr_shift, 'N (T H W) -> N T H W', H=dH, W=dW)

            tgt_masks_matched, ref_masks_matched = [], []
            for (dt, dx, dy), is_inbox_pairwise in zip(self.pixel_offsets, is_inbox_pairwise_st):
                if is_inbox_pairwise.sum() == 0:
                    continue

                src_masks_a = src_masks[..., max(-dy, 0):dH-dy, max(-dx, 0):dW-dx].flatten()
                if dt == 0:
                    # spatial paired pixels
                    src_masks_b = src_masks[..., max(dy, 0):dH+dy, max(dx, 0):dW+dx].flatten()
                else:
                    # temporal paired pixels
                    src_masks_b = src_masks_ctr_shift[..., max(dy, 0):dH+dy, max(dx, 0):dW+dx].flatten()
                tgt_masks_matched.append(src_masks_a[is_inbox_pairwise])
                ref_masks_matched.append(src_masks_b[is_inbox_pairwise])

            if len(tgt_masks_matched) == 0:
                continue

            tgt_masks_matched, ref_masks_matched = torch.cat(tgt_masks_matched), torch.cat(ref_masks_matched)

            # pairwise loss (keep boundary) on the paired pixels with similar lab color and feature correlation,
            # which at least one pixel in the inner box
            tgt_log_fg_prob = F.logsigmoid(tgt_masks_matched)
            tgt_log_bg_prob = F.logsigmoid(-tgt_masks_matched)
            ref_log_fg_prob = F.logsigmoid(ref_masks_matched)
            ref_log_bg_prob = F.logsigmoid(-ref_masks_matched)

            # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
            # we compute the probability in log space to avoid numerical instability
            log_same_fg_prob = tgt_log_fg_prob + ref_log_fg_prob
            log_same_bg_prob = tgt_log_bg_prob + ref_log_bg_prob

            max_log = torch.max(log_same_fg_prob, log_same_bg_prob)
            log_same_prob = torch.log(
                torch.exp(log_same_fg_prob - max_log) +
                torch.exp(log_same_bg_prob - max_log)
            ) + max_log

            losses.append(-log_same_prob.mean())

        return sum(losses) / max(len(losses), 1) if len(losses) > 0 else pred_masks.new_zeros(1)

    def loss_masks_pseudo(self, outputs, targets, indices, num_masks):
        """Compute pixel-wise mask loss with high-quality pseudo instance masks.
        """
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]  # NxTxHpxWp
        tgt_masks_pseudo = torch.cat(
            [t['masks_pseudo'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)  # NxTxHxW
        tgt_mask_scores_pseudo = torch.cat(
            [t['mask_pseudo_scores'][i] for t, (_, i) in zip(targets, indices)]
        ).to(src_masks)  # N

        is_high_conf = tgt_mask_scores_pseudo >= self.pseudo_mask_score_thresh
        src_masks_high_conf = src_masks[is_high_conf]
        tgt_masks_pseudo_high_conf = tgt_masks_pseudo[is_high_conf]
        # No need to align the size between predicted masks and ground-truth masks
        src_masks_high_conf = src_masks_high_conf.flatten(0, 1)[:, None]
        tgt_masks_pseudo_high_conf = tgt_masks_pseudo_high_conf.flatten(0, 1)[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_high_conf,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                tgt_masks_pseudo_high_conf,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks_high_conf,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del src_masks_high_conf
        del tgt_masks_pseudo
        del tgt_masks_pseudo_high_conf

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_spatial_temporal_pairwise_samples(self, targets, mask_features):
        """Compute pairwise affinity by the color similarity and the patch feature correlation.
        """
        if isinstance(mask_features, list):
            H, W = mask_features[-1].shape[-2:]
            mask_features = [F.interpolate(f, (H, W), mode='bilinear', align_corners=False) for f in mask_features]
            mask_features = rearrange(torch.stack(mask_features, dim=1),
                                      '(b t) l c h w -> b t l c h w', b=len(targets))
        else:
            mask_features = mask_features.unsqueeze(2)  # b t 1 c h w

        # mask features: BxTxCxHpxWp, 1/4 resolution of input image
        bs, T, N_l = mask_features.shape[:3]
        H, W = targets[0]["image_lab_color"].shape[-2:]
        if max(H, W) > 480 or N_l > 1:
            H, W = H // 2, W // 2

        # [spatial_size+2×padding−dilation×(kernel_size−1)−1]/stride + 1
        dH = (H + 2 * int(self.boxvis_pairwise_corr_kernel_size / 2) - self.boxvis_pairwise_corr_kernel_size) 
        dH = int(dH / self.boxvis_pairwise_corr_stride) + 1
        dW = (W + 2 * int(self.boxvis_pairwise_corr_kernel_size / 2) - self.boxvis_pairwise_corr_kernel_size)
        dW = int(dW / self.boxvis_pairwise_corr_stride) + 1

        grid_y, grid_x = torch.meshgrid(torch.arange(dH), torch.arange(dW))  # HxW
        grid_indices = (grid_y * dW + grid_x).to(mask_features.device)  # HxW

        for t, features in zip(targets, mask_features):
            if t['masks'].nelement() == 0:
                continue

            tgt_masks = t['masks'].clone()  # NxTxHxW
            tgt_masks = F.interpolate(tgt_masks, (H, W), mode='bilinear', align_corners=False).float()
            tgt_masks = rearrange(self.unfold_pixel(tgt_masks),
                                  'N (T K) (H W) -> N T H W K',
                                  H=dH, W=dW, K=1).gt(0.5).any(dim=-1)

            img_color = t["image_lab_color"]  # Tx3xHxW
            img_color = F.interpolate(img_color, (H, W), mode='bilinear', align_corners=False)  # Tx3xHxW
            img_color = rearrange(self.unfold_pixel(img_color),
                                  'T (C K) (H W) -> T H W K C',
                                  H=dH, W=dW, K=1, C=3)

            if self.enable_patch_corr:
                features = features.flatten(1, 2)
                features = F.interpolate(features, (H, W),
                                         mode='bilinear', align_corners=False)  # TxN_lCxHxW
                features = rearrange(self.unfold_patch(features),
                                     'T (C K) (H W) -> (T H W) K C',
                                     H=dH, W=dW, K=self.boxvis_pairwise_corr_kernel_size ** 2)

            tgt_boxes = t["boxes"]  # NxTx4
            tgt_boxes_c = 0.5 * (tgt_boxes[..., :2] + tgt_boxes[..., 2:]) * \
                          torch.as_tensor([dW, dH], device=tgt_boxes.device).reshape(1, 1, -1)
            ctr_shift_idx = [idx % T for idx in range(1, T+1)]
            dxy_ctrs_temp = (tgt_boxes_c - tgt_boxes_c[:, ctr_shift_idx]).round().long()  # Nx(T-1)x2
            tgt_boxes_exist = ((tgt_boxes[..., 2:] - tgt_boxes[..., :2]) > 0).all(dim=-1)
            tgt_boxes_exist = tgt_boxes_exist & tgt_boxes_exist[:, ctr_shift_idx]  # Nx(T-1)

            N_objs = tgt_boxes.shape[0]
            if self.enable_box_center_shifting:
                # box-center guided shifting between the t and t+1 frames
                indices_ctr_shift = torch.ones_like(tgt_masks).long() * -1
                tgt_masks_ctr_shift = torch.zeros_like(tgt_masks).long() - 1
                img_color_ctr_shift = torch.zeros_like(img_color)[None].repeat(N_objs, 1, 1, 1, 1, 1)
                for obj_i in range(N_objs):
                    for t_i in range(T):
                        dx_c, dy_c = dxy_ctrs_temp[obj_i, t_i]
                        if not tgt_boxes_exist[obj_i, t_i] or dx_c.abs() >= dW-2 or dy_c.abs() >= dH-2:
                            continue

                        sh1, eh1, sw1, ew1 = max(dy_c, 0), min(dH+dy_c, dH), max(dx_c, 0), min(dW+dx_c, dW)
                        sh2, eh2, sw2, ew2 = max(-dy_c, 0), min(dH-dy_c, dH), max(-dx_c, 0), min(dW-dx_c, dW)
                        assert eh1-sh1 == eh2-sh2 and ew1-sw1 == ew2-sw2, 'Check the shifted frame sizes!'

                        indices_ctr_shift[obj_i, t_i, sh1:eh1, sw1:ew1] = grid_indices[sh2:eh2, sw2:ew2] + ctr_shift_idx[t_i] * dH * dW
                        tgt_masks_ctr_shift[obj_i, t_i, sh1:eh1, sw1:ew1] = tgt_masks[obj_i, ctr_shift_idx[t_i], sh2:eh2, sw2:ew2]
                        img_color_ctr_shift[obj_i, t_i, sh1:eh1, sw1:ew1] = img_color[ctr_shift_idx[t_i], sh2:eh2, sw2:ew2]
            else:
                indices_ctr_shift = grid_indices[None, None].repeat(N_objs, T, 1, 1) + \
                                  (torch.as_tensor(ctr_shift_idx, device=grid_indices.device) * dH * dW).reshape(-1, 1, 1)
                tgt_masks_ctr_shift = tgt_masks[:, ctr_shift_idx]
                img_color_ctr_shift = img_color[None, ctr_shift_idx]  # 1xTxHxWxKxC

            grid_indices_objs = grid_indices[None].repeat(T, 1, 1) + \
                                (torch.arange(T) * dH * dW).reshape(-1, 1, 1).to(grid_indices.device)

            is_inbox_pairwise_st = []
            for (dt, dx, dy) in self.pixel_offsets:
                tgt_masks_a = tgt_masks[:, :, max(-dy, 0):dH-dy, max(-dx, 0):dW-dx]
                img_color_a = img_color[None, :, max(-dy, 0):dH-dy, max(-dx, 0):dW-dx]
                if dt == 0:
                    # spatial paired pixels
                    tgt_masks_b = tgt_masks[:, :, max(dy, 0):dH+dy, max(dx, 0):dW+dx]
                    img_color_b = img_color[None, :, max(dy, 0):dH+dy, max(dx, 0):dW+dx]
                else:
                    # temporal paired pixels
                    tgt_masks_b = tgt_masks_ctr_shift[:, :, max(dy, 0):dH+dy, max(dx, 0):dW+dx]
                    img_color_b = img_color_ctr_shift[:, :, max(dy, 0):dH+dy, max(dx, 0):dW+dx]

                # Cond1: inbox, -1 in tgt_masks_b is the padding area after shifting
                if dt == 0:
                    # spatial paired pixels: at least one pixel is in the inner box
                    is_inbox = (tgt_masks_a + tgt_masks_b).flatten(1) > 0  # NxT(Hp-1)(Wp-1)
                else:
                    # temporal paired pixels: both two pixels should be in the inner box
                    is_inbox = (tgt_masks_a.float() + tgt_masks_b.float()).flatten(1) == 2

                # Cond2: high Affinity => sim_color + path_corr
                sim_color = torch.exp(-torch.norm(
                    img_color_a - img_color_b,
                    dim=-1) * 0.5).mean(dim=-1).flatten(1)  # NxT(Hp-1)(Wp-1)
                if sim_color.shape[0] == 1:
                    sim_color = sim_color.repeat(N_objs, 1)
                sim_color = sim_color[is_inbox]

                indices_a = grid_indices_objs[:, max(-dy, 0):dH-dy, max(-dx, 0):dW-dx:].flatten()
                if self.enable_patch_corr:
                    if dt == 0:
                        indices_b = grid_indices_objs[:, max(dy, 0):dH+dy, max(dx, 0):dW+dx].flatten()
                        feat_corr = torch.cat([
                            torch.einsum('nkc,nkc->nk',
                                         F.normalize(features[indices_a[is_inbox_obj]], dim=-1),
                                         F.normalize(features[indices_b[is_inbox_obj]], dim=-1)
                                         ).mean(dim=-1)
                            for is_inbox_obj in is_inbox
                        ])
                    else:
                        indices_ctr_shift_b = indices_ctr_shift[..., max(dy, 0):dH+dy, max(dx, 0):dW+dx].flatten(1)
                        is_inbox = is_inbox & (indices_ctr_shift_b >= 0)
                        feat_corr = torch.cat([
                            torch.einsum('nkc,nkc->nk',
                                         F.normalize(features[indices_a[is_inbox_obj]], dim=-1),
                                         F.normalize(features[indices_b[is_inbox_obj]], dim=-1)
                                         ).mean(dim=-1)
                            for is_inbox_obj, indices_b in zip(is_inbox, indices_ctr_shift_b)
                        ])
                    thresh = self.boxvis_pairwise_color_thresh + 0.5 * self.boxvis_pairwise_corr_thresh
                    is_pairwise = (sim_color + 0.5 * feat_corr) >= thresh

                else:
                    is_pairwise = sim_color >= self.boxvis_pairwise_color_thresh

                is_inbox_pairwise = is_inbox.clone()
                is_inbox_pairwise[is_inbox] = is_pairwise
                is_inbox_pairwise_st.append(is_inbox_pairwise)

            t['st_pair_spatial_size'] = (H, W)
            t['st_pair_unfold_size'] = (dH, dW)
            t['st_pair_indices_ctr_shift'] = indices_ctr_shift
            t['st_pair_is_inbox_pairwise'] = is_inbox_pairwise_st  # n_o [NxT(Hp-1)(Wp-1)]

    def get_loss(self, loss, outputs, targets, indices, num_masks, l_layer=9):
        if self.boxvis_enabled:
            loss_map = {
                'labels': self.loss_labels,
                'masks': self.loss_masks_with_box_supervised,
            }
        else:
            loss_map = {
                'labels': self.loss_labels,
                'masks': self.loss_masks,
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, l_layer)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self._iter += 1
        self.pseudo_mask_score_thresh = 1 / (1 + math.exp(-2 * self._iter / self.max_iters))

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # get pairwise samples in temporal dimension based on patch feature similarity
        if self.boxvis_pairwise_enable:
            with torch.no_grad():
                self._get_spatial_temporal_pairwise_samples(targets, outputs["mask_features"])

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, self.pseudo_mask_score_thresh)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, l_layer=9))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets, self.pseudo_mask_score_thresh)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks, l_layer=i)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
