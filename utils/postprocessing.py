import numpy as np

def get_boxes_v2(output, img_size, anchors):
    """ extract bounding boxes from the last layer (Darknet v2) """
    bias_w, bias_h = anchors
    
    w_img, h_img = img_size[1], img_size[0]
    grid_w, grid_h, num_boxes = output.shape[:3]

    offset_x = \
        np.tile(np.arange(grid_w)[:, np.newaxis], (grid_h, 1, num_boxes))
    offset_y = np.transpose(offset_x, (1, 0, 2))
    boxes = output.copy()
    boxes[:, :, :, 0] = (offset_x + logistic(boxes[:, :, :, 0])) / grid_w
    boxes[:, :, :, 1] = (offset_y + logistic(boxes[:, :, :, 1])) / grid_h
    boxes[:, :, :, 2] = np.exp(boxes[:, :, :, 2]) * bias_w / grid_w
    boxes[:, :, :, 3] = np.exp(boxes[:, :, :, 3]) * bias_h / grid_h

    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img

    return boxes

def parse_yolo_output_v2(output, img_size, num_classes, anchors):
    """ convert the output of the last convolutional layer (Darknet v2) """
    n_coord_box = 4

    num_boxes = output.shape[0] // (n_coord_box + 1 + num_classes)
    output = output.reshape((num_boxes, -1, output.shape[1], output.shape[2]))\
             .transpose((2, 3, 0, 1))
    probs = logistic(output[:, :, :, 4:5]) * softmax(output[:, :, :, 5:], axis=3)
    boxes = get_boxes_v2(output[:, :, :, :4], img_size, anchors)

    return boxes, probs

def get_candidate_objects(output, img_size, classes, anchors, threshold):
    """ convert network output to bounding box predictions """

    #threshold = 0.8
    iou_threshold = 0.4

    boxes, probs = parse_yolo_output_v2(output, img_size, len(classes), anchors)
    filter_mat_probs = (probs >= threshold)
    filter_mat_boxes = np.nonzero(filter_mat_probs)[0:3]
    boxes_filtered = boxes[filter_mat_boxes]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes]

    idx = np.argsort(probs_filtered)[::-1]
    boxes_filtered = boxes_filtered[idx]
    probs_filtered = probs_filtered[idx]
    classes_num_filtered = classes_num_filtered[idx]

    # too many detections - exit
    if len(boxes_filtered) > 1e3:
        print("Too many detections, maybe an error? : {}".format(
            len(boxes_filtered)))
        return []

    probs_filtered = non_maxima_suppression(boxes_filtered, probs_filtered,
                                            classes_num_filtered, iou_threshold)

    filter_iou = (probs_filtered > 0.0)
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for class_id, box, prob in zip(classes_num_filtered, boxes_filtered, probs_filtered):
        result.append([classes[class_id], box[0], box[1], box[2], box[3], prob])

    return result

def logistic(val):
    """ compute the logistic activation """
    return 1.0 / (1.0 + np.exp(-val))

def softmax(val, axis=-1):
    """ compute the softmax of the given tensor, normalizing on axis """
    exp = np.exp(val - np.amax(val, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)

def non_maxima_suppression(boxes, probs, classes_num, thr=0.2):
    """ greedily suppress low-scoring overlapped boxes """
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        for j in range(i+1, len(boxes)):
            if classes_num[i] == classes_num[j] and iou(box, boxes[j]) > thr:
                probs[j] = 0.0

    return probs

def iou(box1, box2, denom="min"):
    """ compute intersection over union score """
    int_tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
             max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    int_lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
             max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

    intersection = max(0.0, int_tb) * max(0.0, int_lr)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
    control_area = min(area1, area2) if denom == "min"  \
                   else area1 + area2 - intersection

    return intersection / control_area