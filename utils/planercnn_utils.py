
import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


############################################################
#  Pytorch Utility Functions
############################################################


def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    ## Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    ## Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])
    ## Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    ## Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    ## Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    ## Box deltas [batch, num_rois, 4]
    deltas = inputs[1]

    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev
    ## Improve performance by trimming to top anchors by score
    ## and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :]
    anchors = anchors[order.data, :]

    ## Apply deltas to anchors to get refined anchors.
    ## [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    ## Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    ## Filter out small boxes
    ## According to Xinlei Chen's paper, this reduces detection accuracy
    ## for small objects, so we're skipping it.

    ## Non-max suppression
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)

    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    ## Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    ## Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    ## Equation 1 in the Feature Pyramid Networks paper. Account for
    ## the fact that our coordinates are normalized here.
    ## e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        ## Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        ## Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        ## Crop and Resize
        ## From Mask R-CNN paper: "We sample four regular locations, so
        ## that we can evaluate either max or average pooling. In fact,
        ## interpolating only a single value at each bin center (without
        ## pooling) is nearly as effective."
        #
        ## Here we use the simplified approach of a single value per bin,
        ## which is how it's done in tf.crop_and_resize()
        ## Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    ## Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)
    ## Pack box_to_level mapping into one array and add another
    ## column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    ## Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def coordinates_roi(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    cooridnates = inputs[1]

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    ind = Variable(torch.zeros(boxes.size()[0]), requires_grad=False).int()
    if boxes.is_cuda:
        ind = ind.cuda()
    cooridnates = cooridnates.unsqueeze(0)  ## CropAndResizeFunction needs batch dimension
    pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(cooridnates, boxes, ind)

    return pooled_features


############################################################
##  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    ## 1. Tile boxes2 and repeate boxes1. This allows us to compare
    ## every boxes1 against every boxes2 without loops.
    ## TF doesn't have an equivalent to np.repeate() so simulate it
    ## using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    ## 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    ## 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    ## 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, gt_parameters, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    ## Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    gt_parameters = gt_parameters.squeeze(0)
    no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
    if config.GPU_COUNT:
        no_crowd_bool = no_crowd_bool.cuda()

    ## Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    ## Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    # print('positive count', positive_roi_bool.sum())

    ## Subsample ROIs. Aim for 33% positive
    ## Positive ROIs
    if positive_roi_bool.sum() > 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data, :]

        ## Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
        roi_gt_parameters = gt_parameters[roi_gt_box_assignment.data]

        ## Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        ## Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data]

        ## Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            ## Transform ROI corrdinates from normalized image space
            ## to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()

        if config.NUM_PARAMETER_CHANNELS > 0:
            masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(
                roi_masks[:, :, :, 0].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = torch.round(masks)
            parameters = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(
                roi_masks[:, :, :, 1].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = torch.stack([masks, parameters], dim=-1)
        else:
            masks = Variable(
                CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes,
                                                                                     box_ids).data,
                requires_grad=False).squeeze(1)
            masks = torch.round(masks)
            pass

        ## Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        ## binary cross entropy loss.
    else:
        positive_count = 0

    ## 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    ## Negative ROIs. Add enough to maintain positive:negative ratio.
    if (negative_roi_bool > 0).sum() > 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # print('count', positive_count, negative_count)
    # print(roi_gt_class_ids)

    ## Append negative ROIs and pad bbox deltas and masks that
    ## are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        if config.NUM_PARAMETER_CHANNELS > 0:
            zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], 2),
                             requires_grad=False)
        else:
            zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                             requires_grad=False)
            pass
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_parameters = torch.cat([roi_gt_parameters, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_parameters = torch.cat([roi_gt_parameters, zeros], dim=0)
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_parameters = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
            roi_gt_parameters = roi_gt_parameters.cuda()
            pass

    return rois, roi_gt_class_ids, deltas, masks, roi_gt_parameters


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[2])), boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])), boxes[:, 3].clamp(float(window[1]), float(window[3]))],
        dim=-1)
    return boxes


def refine_detections(rois, probs, deltas, parameters, window, config, return_indices=False, use_nms=1, one_hot=True):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    ## Class IDs per ROI

    if len(probs.shape) == 1:
        class_ids = probs.long()
    else:
        _, class_ids = torch.max(probs, dim=1)
        pass

    ## Class probability of the top class of each ROI
    ## Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()

    if len(probs.shape) == 1:
        class_scores = torch.ones(class_ids.shape)
        deltas_specific = deltas
        class_parameters = parameters
        if config.GPU_COUNT:
            class_scores = class_scores.cuda()
    else:
        class_scores = probs[idx, class_ids.data]
        deltas_specific = deltas[idx, class_ids.data]
        class_parameters = parameters[idx, class_ids.data]
    ## Apply bounding box deltas
    ## Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()

    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    ## Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois = refined_rois * scale
    ## Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    ## Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    ## TODO: Filter out boxes with zero area

    ## Filter out background boxes
    keep_bool = class_ids > 0

    ## Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE and False:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

    keep_bool = keep_bool & (refined_rois[:, 2] > refined_rois[:, 0]) & (refined_rois[:, 3] > refined_rois[:, 1])

    if keep_bool.sum() == 0:
        if return_indices:
            return torch.zeros((0, 10)).cuda(), torch.zeros(0).long().cuda(), torch.zeros((0, 4)).cuda()
        else:
            return torch.zeros((0, 10)).cuda()
        pass

    keep = torch.nonzero(keep_bool)[:, 0]

    if use_nms == 2:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        ixs = torch.arange(len(pre_nms_class_ids)).long().cuda()
        ## Sort
        ix_rois = pre_nms_rois
        ix_scores = pre_nms_scores
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data, :]

        nms_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        nms_keep = keep[ixs[order[nms_keep].data].data]
        keep = intersect1d(keep, nms_keep)
    elif use_nms == 1:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            ## Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

            ## Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data, :]

            class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

            ## Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i == 0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)
    else:
        pass

    ## Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    # print('num detectinos', len(keep))

    ### Apply plane anchors
    class_parameters = config.applyAnchorsTensor(class_ids, class_parameters)
    ## Arrange output as [N, (y1, x1, y2, x2, class_id, score, parameters)]
    ## Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1),
                        class_parameters[keep.data]), dim=1)

    if return_indices:
        ori_rois = rois * scale
        ori_rois = clip_to_window(window, ori_rois)
        ori_rois = torch.round(ori_rois)
        ori_rois = ori_rois[keep.data]
        return result, keep.data, ori_rois

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, mrcnn_parameter, image_meta, return_indices=False, use_nms=1,
                    one_hot=True):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    ## Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    if len(mrcnn_class) == 0:
        if return_indices:
            return torch.zeros(0), torch.zeros(0), torch.zeros(0)
        else:
            return torch.zeros(0)

    return refine_detections(rois, mrcnn_class, mrcnn_bbox, mrcnn_parameter, window, config,
                             return_indices=return_indices, use_nms=use_nms, one_hot=one_hot)


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    ## Positive and Negative anchors contribute to the loss,
    ## but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    ## Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:, 0], indices.data[:, 1], :]
    anchor_class = anchor_class[indices.data[:, 0], indices.data[:, 1]]

    ## Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Positive anchors contribute to the loss, but negative and
    ## neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)
    ## Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:, 0], indices.data[:, 1]]

    ## Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

    ## Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    ## Loss
    if len(target_class_ids) > 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0].data, :]
        pred_bbox = pred_bbox[indices[:, 0].data, indices[:, 1].data, :]

        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(config, target_masks, target_class_ids, target_parameters, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]

        if config.GLOBAL_MASK:
            y_pred = pred_masks[indices[:, 0], 0, :, :]
        else:
            y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]
            pass

        if config.NUM_PARAMETER_CHANNELS == 1:
            if config.OCCLUSION:
                visible_pred = pred_masks[indices[:, 0], -1, :, :]
                visible_gt = y_true[:, :, :, -1]
                y_true = y_true[:, :, :, 0]
                loss = F.binary_cross_entropy(y_pred, y_true) + F.binary_cross_entropy(visible_pred, visible_gt)
            else:
                depth_pred = pred_masks[indices[:, 0], -1, :, :]
                depth_gt = y_true[:, :, :, -1]
                y_true = y_true[:, :, :, 0]
                loss = F.binary_cross_entropy(y_pred, y_true) + l1LossMask(depth_pred, depth_gt,
                                                                           (depth_gt > 1e-4).float())
                pass
        elif config.NUM_PARAMETER_CHANNELS == 4:
            depth_pred = pred_masks[indices[:, 0], -config.NUM_PARAMETER_CHANNELS, :, :]
            depth_gt = y_true[:, :, :, -1]
            y_true = y_true[:, :, :, 0]
            normal_pred = pred_masks[indices[:, 0], -(config.NUM_PARAMETER_CHANNELS - 1):, :, :]
            normal_gt = target_parameters[indices[:, 0]]
            normal_gt = normal_gt / torch.clamp(torch.norm(normal_gt, dim=-1, keepdim=True), min=1e-4)
            loss = F.binary_cross_entropy(y_pred, y_true) + l1LossMask(depth_pred, depth_gt,
                                                                       (depth_gt > 1e-4).float()) + l2NormLossMask(
                normal_pred, normal_gt.unsqueeze(-1).unsqueeze(-1), y_true, dim=1)
        else:
            ## Binary cross entropy
            loss = F.binary_cross_entropy(y_pred, y_true)
            pass
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_parameter_loss(target_parameters, target_class_ids, pred_parameters):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_parameters = target_parameters[indices[:, 0].data, :]
        pred_parameters = pred_parameters[indices[:, 0].data, indices[:, 1].data, :]
        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_parameters, target_parameters)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                   target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(config, target_mask, target_class_ids, target_parameters, mrcnn_mask)
    mrcnn_parameter_loss = compute_mrcnn_parameter_loss(target_parameters, target_class_ids, mrcnn_parameters)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss]

###############################3
# From planarcnn utils

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    ## Anchors
    ## [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    anchors = np.concatenate(anchors, axis=0)
    return anchors



# From models
############################################################
#  Data Formatting
############################################################


def mold_inputs(config, images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        ## Resize image to fit the model expected size
        ## TODO: move resizing to mold_image()
        molded_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        molded_image = mold_image(molded_image, config)
        ## Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, window,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        ## Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    ## Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows


def unmold_detections(config, detections, mrcnn_mask, image_shape, window, debug=False):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is
            excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    ## How many detections do we have?
    ## Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    ## Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    parameters = detections[:N, 6:]
    if config.GLOBAL_MASK:
        masks = mrcnn_mask[np.arange(N), :, :, 0]
    else:
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        pass

    ## Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  ## y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    ## Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

    if debug:
        print(masks.shape, boxes.shape)
        for maskIndex, mask in enumerate(masks):
            print(maskIndex, boxes[maskIndex].astype(np.int32))
            cv2.imwrite('test/local_mask_' + str(maskIndex) + '.png', (mask * 255).astype(np.uint8))
            continue

    ## Filter out detections with zero area. Often only happens in early
    ## stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        parameters = np.delete(parameters, exclude_ix, axis=0)
        N = class_ids.shape[0]

    ## Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        ## Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1) \
        if full_masks else np.empty((0,) + masks.shape[1:3])

    if debug:
        print(full_masks.shape)
        for maskIndex in range(full_masks.shape[2]):
            cv2.imwrite('test/full_mask_' + str(maskIndex) + '.png',
                        (full_masks[:, :, maskIndex] * 255).astype(np.uint8))
            continue
        pass
    return boxes, class_ids, scores, full_masks, parameters



# From utils
############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    ## Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    ## Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    ## Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    ## Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    ## Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    ## Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes
