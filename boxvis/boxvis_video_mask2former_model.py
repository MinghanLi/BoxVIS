import copy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color
from scipy.optimize import linear_sum_assignment

import numpy as np
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.utils.box_ops import box_xyxy_to_cxcywh
from boxvis.modeling.video_criterion import BoxVISVideoSetCriterion, BoxVISTeacherSetPseudoMask
from boxvis.modeling.video_matcher import BoxVISVideoHungarianMatcher
from boxvis.data.datasets.bvisd import BVISD_TO_YTVIS_2021, YTVIS_2021_TO_BVISD, BVISD_TO_OVIS, OVIS_TO_BVISD
from boxvis.mdqe_overtracker_efficient import Clips, MDQE_OverTrackerEfficient


def convert_box_to_mask(outputs_box: torch.Tensor, h: int, w: int):
    box_normalizer = torch.as_tensor([w, h, w, h], dtype=outputs_box.dtype,
                                     device=outputs_box.device).reshape(1, 1, -1)
    outputs_box_wonorm = outputs_box * box_normalizer  # B, Q, 4
    outputs_box_wonorm = torch.cat([outputs_box_wonorm[..., :2].floor(),
                                    outputs_box_wonorm[..., 2:].ceil()], dim=-1)
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=outputs_box.device),
                                    torch.arange(w, device=outputs_box.device))  # H, W
    grid_y = grid_y.reshape(1, 1, h, w)
    grid_x = grid_x.reshape(1, 1, h, w)

    # repeat operation will greatly expand the computational graph
    gt_x1 = grid_x > outputs_box_wonorm[..., 0, None, None]
    lt_x2 = grid_x <= outputs_box_wonorm[..., 2, None, None]
    gt_y1 = grid_y > outputs_box_wonorm[..., 1, None, None]
    lt_y2 = grid_y <= outputs_box_wonorm[..., 3, None, None]
    cropped_box_mask = gt_x1 & lt_x2 & gt_y1 & lt_y2

    return cropped_box_mask


@META_ARCH_REGISTRY.register()
class BoxVIS_VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        hidden_dim: int,
        num_queries: int,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        num_classes: int,
        # boxvis
        gen_pseudo_mask: nn.Module,
        boxvis_enabled: bool,
        boxvis_ema_enabled: bool,
        boxvis_bvisd_enabled: bool,
        data_name: str,
        # inference
        tracker_type: str,
        window_inference: bool,
        test_topk_per_image: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        merge_on_cpu: bool,
        # tracking
        num_max_inst_test: int,
        num_frames_test: int,
        num_frames_window_test: int,
        clip_stride: int,

    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            boxvis_enabled: if True, use only box-level annotation; otherwise pixel-wise annotations
            boxvis_ema_enabled: if True, use Teacher Net to produce high-quality pseudo masks
            boxvis_bvisd_enabled: if True, use box-supervised VIS dataset (BVISD), including
                pseudo video clip from COCO, videos from YTVIS21 and OVIS.
        """
        super().__init__()

        # Student Net
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_coco = data_name.startswith("coco")

        # boxvis
        if boxvis_enabled and boxvis_ema_enabled:
            # Teacher Net
            self.backbone_t = copy.deepcopy(backbone)
            self.sem_seg_head_t = copy.deepcopy(sem_seg_head)
            self.gen_pseudo_mask = gen_pseudo_mask
            self.backbone_t.requires_grad_(False)
            self.sem_seg_head_t.requires_grad_(False)
            self.ema_shadow_decay = 0.999

        self.boxvis_enabled = boxvis_enabled
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.boxvis_bvisd_enabled = boxvis_bvisd_enabled
        self.data_name = data_name
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"

        # additional args reference
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.window_inference = window_inference
        self.test_topk_per_image = test_topk_per_image
        self.merge_on_cpu = merge_on_cpu
        
        # clip-by-clip tracking
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_test = num_frames_test
        self.num_frames_window_test = num_frames_window_test
        self.clip_stride = clip_stride

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        proj_weight = dice_weight
        pair_weight = 1.
        if cfg.MODEL.BoxVIS.BoxVIS_ENABLED:
            dice_weight, mask_weight = 0.5*dice_weight, 0.5*mask_weight
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                           "loss_mask_proj": proj_weight, "loss_mask_pair": pair_weight}
        else:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        # building criterion
        matcher = BoxVISVideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_proj=proj_weight,
            cost_pair=pair_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
        )

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.BoxVIS.EMA_ENABLED:
            gen_pseudo_mask = BoxVISTeacherSetPseudoMask(matcher)
        else:
            gen_pseudo_mask = None

        losses = ["labels", "masks"]
        criterion = BoxVISVideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            is_coco=cfg.DATASETS.TEST[0].startswith("coco"),
            # boxvis parameters
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_pairwise_enable=cfg.MODEL.BoxVIS.PAIRWISE_ENABLED,
            boxvis_pairwise_num_stpair=cfg.MODEL.BoxVIS.PAIRWISE_STPAIR_NUM,
            boxvis_pairwise_dilation=cfg.MODEL.BoxVIS.PAIRWISE_DILATION,
            boxvis_pairwise_color_thresh=cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH,
            boxvis_pairwise_corr_kernel_size=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_KERNEL_SIZE,
            boxvis_pairwise_corr_stride=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_STRIDE,
            boxvis_pairwise_corr_thresh=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_THRESH,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
            boxvis_pseudo_mask_score_thresh=cfg.MODEL.BoxVIS.PSEUDO_MASK_SCORE_THRESH,
            max_iters=cfg.SOLVER.MAX_ITER,
        )
        num_classes = sem_seg_head.num_classes

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            # boxvis
            "gen_pseudo_mask": gen_pseudo_mask,
            'boxvis_enabled': cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            "boxvis_ema_enabled": cfg.MODEL.BoxVIS.EMA_ENABLED,
            "boxvis_bvisd_enabled": cfg.MODEL.BoxVIS.BVISD_ENABLED,
            "data_name": cfg.DATASETS.TEST[0],
            # inference
            "tracker_type": cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE,
            "window_inference": cfg.MODEL.BoxVIS.TEST.WINDOW_INFERENCE,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "is_multi_cls": cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES,
            "merge_on_cpu": cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU,
            # tracking
            "num_max_inst_test": cfg.MODEL.BoxVIS.TEST.NUM_MAX_INST,
            "num_frames_test": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES,
            "num_frames_window_test": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW,
            "clip_stride": cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]: each dict has the results for one image.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_norm = ImageList.from_tensors(images_norm, self.size_divisibility)
        images = ImageList.from_tensors(images, self.size_divisibility)

        if self.training and self.boxvis_ema_enabled:
            # ---------------- prepare EMA for Teacher net ---------------------
            backbone_shadow, sem_seg_head_shadow = {}, {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    backbone_shadow[name] = param.data.clone().detach()

            for name, param in self.sem_seg_head.named_parameters():
                if param.requires_grad:
                    sem_seg_head_shadow[name] = param.data.clone().detach()

            w_shadow = 1.0 - self.ema_shadow_decay
            # apply weighted weights to the teacher net
            for name, param in self.backbone_t.named_parameters():
                if name in backbone_shadow:
                    param.data = w_shadow * backbone_shadow[name] + (1-w_shadow) * param.data
            for name, param in self.sem_seg_head_t.named_parameters():
                if name in sem_seg_head_shadow:
                    param.data = w_shadow * sem_seg_head_shadow[name] + (1-w_shadow) * param.data

        if self.training:
            features_s = self.backbone(images_norm.tensor)
            outputs_s = self.sem_seg_head(features_s)
            targets = self.prepare_targets(batched_inputs, images)

            if self.boxvis_ema_enabled:
                # ------------------ Teacher Net -----------------------------
                features_t = self.backbone_t(images_norm.tensor)
                outputs_t = self.sem_seg_head_t(features_t)
                # generate pseudo masks via teacher outputs
                targets = self.gen_pseudo_mask(outputs_t, targets)

            losses = self.criterion(outputs_s, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

        else:
            
            # NOTE consider only B=1 case.
            if self.tracker_type == 'minvis':
                outputs_s = self.run_window_inference(images_norm.tensor)
                outputs_s = self.post_processing(outputs_s)
                return self.inference_video(batched_inputs, images_norm, outputs_s)

            elif self.tracker_type == 'mdqe':
                return self.run_window_inference_mdqe(batched_inputs, images_norm)
            else:
                raise ValueError('the type of tracker only supports {minvis, mdqe}.')

    def prepare_targets(self, targets, images):
        # Note: images without MEAN and STD normalization
        BT, c, h_pad, w_pad = images.tensor.shape
        box_normalizer = torch.as_tensor([w_pad, h_pad, w_pad, h_pad],
                                         dtype=torch.float32, device=self.device).reshape(1, -1)

        clip_gt_instances = []
        images = images.tensor.reshape(BT//self.num_frames, self.num_frames, -1, h_pad, w_pad)
        for targets_per_video, images_per_video in zip(targets, images):
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            gt_boxes_per_video = torch.zeros([_num_instance, self.num_frames, 4], dtype=torch.float32, device=self.device)
            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)

            gt_ids_per_video, images_lab_color = [], []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                _update_box = box_xyxy_to_cxcywh(targets_per_frame.gt_boxes.tensor)[..., 2:].gt(0).all(-1)
                gt_boxes_per_video[_update_box, f_i] = targets_per_frame.gt_boxes.tensor[_update_box] / box_normalizer  # xyxy
                if self.boxvis_enabled:
                    gt_masks_per_video[:, f_i] = convert_box_to_mask(gt_boxes_per_video[:, f_i], h_pad, w_pad)
                    # Note: check color channels should be rgb, which is controlled by cfg.INPUT.FORMAT
                    # Note: images_per_video without MEAN and STD normalization.
                    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html
                    images_lab_color.append(color.rgb_to_lab((images_per_video[f_i] / 255.).unsqueeze(0)))  # 1x3xHxW
                else:
                    if isinstance(targets_per_frame.gt_masks, BitMasks):
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                    else:  # polygon
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long()  # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()          # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()     # N, num_frames, H, W
            gt_boxes_per_video = gt_boxes_per_video[valid_bool_clip].float()     # N, num_frames, 4
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id  # obj id mapping

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video,
                    "masks": gt_masks_per_video, "boxes": gt_boxes_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"]
                }
            )
            if self.boxvis_enabled:
                clip_gt_instances[-1]["image_lab_color"] = torch.cat(images_lab_color)  # num_frames, 3, H, W

        return clip_gt_instances

    def run_window_inference(self, images_tensor):
        out_list = []
        start_idx_window, end_idx_window = 0, 0
        for i in range(len(images_tensor)):
            if i + self.num_frames_test > len(images_tensor):
                break

            if i + self.num_frames_test > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = self.backbone(images_tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames_test]
                        for k, v in features_window.items()}
            out = self.sem_seg_head(features)
            del out['aux_outputs']
            if self.merge_on_cpu:
                out = {k: v.cpu() for k, v in out.items()}
            out_list.append(out)

        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'].float() for x in out_list]).detach()  # (V-t+1)xQxK
        outputs['pred_masks'] = torch.cat([x['pred_masks'].float() for x in out_list]).detach()  # (V-t+1)xQxtxHxW
        outputs['pred_embds'] = torch.cat([x['pred_embds'].float() for x in out_list]).detach()  # (V-t+1)xQxC

        return outputs

    def post_processing(self, outputs):
        n_clips, q, n_t, h, w = outputs['pred_masks'].shape
        pred_logits = list(torch.unbind(outputs['pred_logits']))  # (V-t+1) q k
        pred_masks = list(torch.unbind(outputs['pred_masks']))    # (V-t+1) q t h w
        pred_embds = list(torch.unbind(outputs['pred_embds']))    # (V-t+1) q c

        out_logits = [pred_logits[0]]
        out_masks = [pred_masks[0]]
        out_embds = [pred_embds[0]]
        for i in range(1, len(pred_logits)):
            mem_embds = torch.stack(out_embds[-2:]).mean(dim=0)
            indices = self.match_from_embds(mem_embds, pred_embds[i])
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits) / len(out_logits)
        out_masks_mean = []
        for v in range(n_clips+n_t-1):
            n_t_valid = min(v+1, n_t)
            m = []
            for t in range(n_t_valid):
                if v-t < n_clips:
                    m.append(out_masks[v-t][:, t])  # q, h, w
            out_masks_mean.append(torch.stack(m).mean(dim=0))  # q, h, w

        outputs['pred_masks'] = torch.stack(out_masks_mean, dim=1)  # t * [q h w] -> q t h w
        outputs['pred_scores'] = F.softmax(out_logits, dim=-1)[:, :-1]  # q k+1

        return outputs

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def inference_video(self, batched_inputs, images, outputs):
        mask_scores = outputs["pred_scores"]  # cQ, K+1
        mask_pred = outputs["pred_masks"]  # cQ, V, H, W or [V/C * (cQ, C, H, W)]

        # upsample masks
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])

        num_topk = max(int(mask_scores.gt(0.05).sum()), self.test_topk_per_image)
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        scores_per_video, topk_indices = mask_scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')

        mask_pred = mask_pred[topk_indices]
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, t, H, W
        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

        mask_quality_scores = (mask_pred > 1).flatten(1).sum(1) / (mask_pred > -1).flatten(1).sum(1).clamp(min=1)
        scores_per_video = scores_per_video * mask_quality_scores

        masks_per_video = []
        for m in mask_pred:
            # slower speed but memory efficiently for long videos
            m = retry_if_cuda_oom(F.interpolate)(
                m.unsqueeze(0),
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0) > 0.
            masks_per_video.append(m.cpu())

        scores_per_video = scores_per_video.tolist()
        labels_per_video = labels_per_video.tolist()

        scores_per_video, labels_per_video, masks_per_video = self.bvisd_mapping_category_ids(
            scores_per_video, labels_per_video, masks_per_video
        )

        processed_results = {
            "image_size": (out_height, out_width),
            "pred_scores": scores_per_video,
            "pred_labels": labels_per_video,
            "pred_masks": masks_per_video,
        }

        return processed_results

    def run_window_inference_mdqe(self, batched_inputs, images):
        video_len = len(images.tensor)
        self.num_frames_window_test = video_len if self.data_name == 'ytvis_2021' else self.num_frames_window_test

        # OverTracker is memory-friendly, processing instance segmentation (long) clip by (long) clip
        # where the length of long clip is controlled by self.num_frames_window_test
        merge_device = "cpu" if self.merge_on_cpu else self.device
        tracker = MDQE_OverTrackerEfficient(
            video_len, self.num_classes, self.num_max_inst_test, self.num_frames_test, self.num_frames_window_test,
            self.clip_stride, self.hidden_dim, self.apply_cls_thres, merge_device, self.data_name
        )

        results_window_list = []
        start_idx_window, end_idx_window = 0, 0
        for i in range(video_len):
            is_first_clip = i == 0
            is_last_clip = (i + self.num_frames_test) == (video_len - 1)
            if i + self.num_frames_test > video_len:
                break

            if i + self.num_frames_test > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = self.backbone(images.tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames_test]
                        for k, v in features_window.items()}
            outputs = self.sem_seg_head(features)
            del outputs['aux_outputs']
            if self.merge_on_cpu:
                outputs = {k: v.cpu() for k, v in outputs.items()}

            i_map_to_0 = i % self.num_frames_window_test
            frame_idx_clip = list(range(i_map_to_0, i_map_to_0+self.num_frames_test))
            outputs_per_window = self.post_processing_mdqe_overtracker(
                tracker, outputs, frame_idx_clip, is_first_clip, is_last_clip
            )
            if outputs_per_window is not None:
                results_per_window = self.inference_video_window_mdqe(
                    i, batched_inputs, images, outputs_per_window, is_last_clip
                )
                results_window_list.append(results_per_window)
        
        return self.inference_video_mdqe(batched_inputs, results_window_list)

    def post_processing_mdqe_overtracker(self, tracker, outputs, frame_idx_clip, is_first_clip=False, is_last_clip=False):
        q, n_t, h, w = outputs['pred_masks'][0].shape

        pred_cls_probs = F.softmax(outputs['pred_logits'][0].float(), dim=-1)[..., :-1]  # q k
        pred_masks = outputs['pred_masks'][0].float()  # q t h w
        pred_embds = outputs['pred_embds'][0].float()  # q c

        mask_quality_scores = (pred_masks > 1).flatten(1).sum(1) / (pred_masks > -1).flatten(1).sum(1).clamp(min=1)
        pred_cls_probs = pred_cls_probs * mask_quality_scores.reshape(-1, 1)

        scores, classes = pred_cls_probs.max(dim=-1)
        sorted_scores, sorted_idx = scores.sort(descending=True)
        valid_idx = sorted_idx[:max(1, int((sorted_scores > self.apply_cls_thres).sum()))]

        # Non-maximum suppression (NMS) based on mask IoU
        m_soft = pred_masks[valid_idx].sigmoid().flatten(1)
        numerator = m_soft[None] * m_soft[:, None].gt(0.5)
        denominator = m_soft[None] + m_soft[:, None].gt(0.5) - numerator
        siou = numerator.sum(dim=-1) / denominator.sum(dim=-1)
        max_siou = torch.triu(siou, diagonal=1).max(dim=0)[0]
        valid_idx = valid_idx[max_siou < 0.75]
        sim = torch.mm(
            F.normalize(pred_embds[valid_idx]),
            F.normalize(pred_embds[valid_idx]).t(),
        )
        max_sim = torch.triu(sim, diagonal=1).max(dim=0)[0]
        valid_idx = valid_idx[max_sim < 0.85]

        clip_results = Clips((h, w), frame_idx_clip)
        clip_results.scores = scores[valid_idx]
        clip_results.classes = classes[valid_idx]
        clip_results.cls_probs = pred_cls_probs[valid_idx]
        clip_results.mask_logits = pred_masks[valid_idx]
        clip_results.query_embeds = pred_embds[valid_idx]

        # update the current clip
        tracker.update(clip_results, is_first_clip=is_first_clip)

        # Save output results clip by clip, which is memory-friendly. After inference of the video,
        # the instance masks of all clips will be directly merged into the .json file (mdqe/data/ytvis_eval.py).
        is_output = ((frame_idx_clip[0] + 1) % self.num_frames_window_test) == 0
        if is_last_clip or is_output:
            return tracker.get_result(is_last_clip=is_last_clip)
        else:
            return None

    def inference_video_window_mdqe(self, cur_frame_idx, batched_inputs, images, outputs_window, is_last_clip):
        pred_masks = outputs_window["pred_masks"].cpu()              # cQ x T_w x H x W
        pred_scores = outputs_window["pred_cls_scores"].cpu()        # cQ x K
        pred_obj_ids = outputs_window["obj_ids"]                     # cQ, gt_ids for sot task

        # upsample masks
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])

        pred_masks = retry_if_cuda_oom(F.interpolate)(
            pred_masks,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, T_w, H, W
        pred_masks = pred_masks[:, :, : image_size[0], : image_size[1]]

        mask_quality_scores = (pred_masks.sigmoid() > 0.75).flatten(1).sum(1) / \
                              (pred_masks.sigmoid() > 0.5).flatten(1).sum(1).clamp(min=1)
        pred_scores = pred_scores * mask_quality_scores.reshape(-1, 1)

        pred_masks_list = []
        num_nonblank_masks_list = []
        for m in pred_masks:
            # slower speed but memory efficiently for long videos
            m = retry_if_cuda_oom(F.interpolate)(
                m.unsqueeze(0),
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0) > 0.
            pred_masks_list.append(m.cpu())
            num_nonblank_masks_list.append((m.sum((-2, -1)) > 0).sum())

        results_per_window_list = []
        for obj_id, s, m, nonblank in zip(pred_obj_ids, pred_scores, pred_masks_list, num_nonblank_masks_list):
            segms = [
                mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
                for _mask in m
            ]
            for rle in segms:
                rle["counts"] = rle["counts"].decode("utf-8")

            frame_id_start = cur_frame_idx + 1 - len(segms) if not is_last_clip \
                else cur_frame_idx + self.num_frames_test - len(segms)
            res = {
                "obj_id": int(obj_id),
                "score": np.array(s),
                "segmentations": segms,
                "num_nonblank_masks": np.array(nonblank),
                "frame_id_start": frame_id_start
            }
            results_per_window_list.append(res)

        return results_per_window_list
    
    def inference_video_mdqe(self, batched_inputs, results_window_list):
        assert len(batched_inputs) == 1, "More than one inputs are loaded for inference!"

        video_id = int(batched_inputs[0]["video_id"])
        video_len = int(batched_inputs[0]["video_len"])
        height = int(batched_inputs[0]["height"])
        width = int(batched_inputs[0]["width"])
    
        blank_rle_mask = mask_util.encode(np.zeros((height, width, 1), order="F", dtype="uint8"))[0]
        blank_rle_mask["counts"] = blank_rle_mask["counts"].decode("utf-8")
    
        ytvis_results = []
    
        num_objs = 0
        obj_ids = set([res["obj_id"] for res in sum(results_window_list, [])])
        for obj_id in obj_ids:
            obj_dict = {
                "video_id": video_id,
                "obj_id": obj_id,
                "score": [],
                "segmentations": [blank_rle_mask] * video_len,
                "num_nonblank_masks": []
            }
    
            for w_i, results in enumerate(results_window_list):
                if len(results) == 0:
                    continue
    
                for res in results:
                    if res["obj_id"] != obj_id:
                        continue
    
                    # K, class scores
                    obj_dict["score"].append(res["score"])
                    # List with T frames, where masks have been encoded
                    f_id_s = res["frame_id_start"]
                    f_id_e = f_id_s + len(res["segmentations"])
                    obj_dict["segmentations"][f_id_s:f_id_e] = res["segmentations"]
                    obj_dict["num_nonblank_masks"].append(res["num_nonblank_masks"])
    
            assert len(obj_dict["segmentations"]) == video_len, \
                f'The video has {video_len} frames, but the prediction has {len(obj_dict["segmentations"])} frames!'
    
            assert len(obj_dict["score"])
            nonblank_masks_ratio = sum(obj_dict["num_nonblank_masks"]) / video_len
            nonblank_scores = 1 - np.exp(-3 * nonblank_masks_ratio)
            scores = sum(obj_dict["score"]) / len(obj_dict["score"])
            classes = np.arange(len(scores))[scores >= min(self.apply_cls_thres, scores.min())]
            # print('predicted label and score: ', scores.argmax(), scores.max())
    
            for c in classes:
                scores_per_video = [float(scores[c]) * nonblank_scores]
                labels_per_video = [int(c)]
                segms_per_video = [obj_dict["segmentations"]]
                scores_per_video, labels_per_video, segms_per_video = self.bvisd_mapping_category_ids(
                    scores_per_video, labels_per_video, segms_per_video
                )

                for s, l, segm in zip(scores_per_video, labels_per_video, segms_per_video):
                    ytvis_results.append({
                        "video_id": obj_dict["video_id"],
                        "score": s,
                        "category_id": l,
                        "segmentations": segm,
                        "height": height,
                        "width": width
                    })
                    num_objs += 1
    
        return ytvis_results

    def bvisd_mapping_category_ids(self, scores, labels, masks):
        # map tgt labels to src labels: joint training, but isolated inference
        if self.boxvis_bvisd_enabled and not self.data_name.startswith('bvisd'):
            if self.data_name.startswith('ytvis_2021'):
                tgt2src = BVISD_TO_YTVIS_2021
            elif self.data_name.startswith('ovis'):
                tgt2src = BVISD_TO_OVIS
            else:
                raise ValueError('only support YTVIS21 or OVIS datasets')

            src_labels, src_scores, src_masks = [], [], []
            for i, (tgt_score, tgt_label, tgt_mask) in enumerate(zip(scores, labels, masks)):
                tgt_label = tgt_label + 1
                if tgt_label in tgt2src:
                    src_label_list = tgt2src[tgt_label]
                    if not isinstance(src_label_list, list):
                        # a tgt label is corresponding to 2 src labels, (bus, car -> vehical in OVIS)
                        src_label_list = [src_label_list]

                    # TODO: ensure to merge similar classes in YTVIS21: car, truck -> vehicle?
                    src_label_list = src_label_list[:1] if self.data_name[-9:] == 'dev_merge' else src_label_list
                    for src_label in src_label_list:
                        src_labels.append(src_label - 1)
                        src_scores.append(tgt_score)
                        src_masks.append(tgt_mask)

            return src_scores, src_labels, src_masks

        # map src labels to tgt labels: isolated training, but inference on bvisd
        if not self.boxvis_bvisd_enabled and self.data_name.startswith('bvisd'):
            src_type = 'ytvis_2021'
            if src_type.startswith('ytvis_2021'):
                tgt2src = YTVIS_2021_TO_BVISD
            elif src_type.startswith('ovis'):
                tgt2src = OVIS_TO_BVISD
            else:
                raise ValueError('only support YTVIS21 or OVIS datasets')

            src_labels, src_scores, src_masks = [], [], []
            for i, (tgt_score, tgt_label, tgt_mask) in enumerate(zip(scores, labels, masks)):
                tgt_label = tgt_label + 1
                if tgt_label in tgt2src:
                    src_label = tgt2src[tgt_label]

                    src_labels.append(src_label - 1)
                    src_scores.append(tgt_score)
                    src_masks.append(tgt_mask)

            return src_scores, src_labels, src_masks

        # isolated training, but joint inference
        if not self.boxvis_bvisd_enabled and self.data_name[-9:] == 'dev_merge':
            if self.data_name.startswith('ytvis_2021'):
                for i, label in enumerate(labels):
                    if label == 36:
                        # 5 car, 37 truck -> 5 vehicle
                        labels[i] = 4
            elif self.data_name.startswith('ovis'):
                for i, label in enumerate(labels):
                    if label == 21:
                        # 21 "Motorcycle" & 22 "Bicycle" -> 21 motorbike
                        labels[i] = 20
            else:
                raise ValueError('only support YTVIS21 or OVIS datasets')

            return scores, labels, masks



