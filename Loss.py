import tensorflow as tf



def IOU(box1, box2):
    x1 = tf.reduce_max(tf.convert_to_tensor([box1[0], box2[0]]))
    y1 = tf.reduce_max(tf.convert_to_tensor([box1[1], box2[1]]))
    x2 = tf.reduce_min(tf.convert_to_tensor([box1[2], box2[2]]))
    y2 = tf.reduce_min(tf.convert_to_tensor([box1[3], box2[3]]))

    intersection = tf.reduce_max(tf.convert_to_tensor([0, x2 - x1])) * tf.reduce_max(tf.convert_to_tensor([0, y2 - y1]))

    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

    iou = intersection/union

    return iou

def Model_Loss(num_classes, bboxes):
    @tf.function
    def Loss(Model_Output, Target_Output):
        Target_Output = tf.constant(Target_Output)
        loss = tf.float32(0)
        lambda_coord = tf.float32(5)
        lambda_noobj = tf.float32(5)


        section = Model_Output[row, point]
        box_x, box_y, box_w, box_h, objectness, class_prob = tf.split(section, [bboxes, bboxes, bboxes, bboxes, bboxes, num_classes], axis=-1)

        box_x = tf.sigmoid(box_x)
        box_y = tf.sigmoid(box_y)
        box_w = tf.exp(box_w)
        box_h = tf.exp(box_h)
        objectness = tf.sigmoid(objectness)
        class_prob = tf.sigmoid(class_prob)


        target_x, target_y, target_w, target_h, target_class = tf.split(Target_Output, [1, 1, 1, 1, num_classes], axis=-1)
        

        object_appears_mask = tf.zeros_like(class_prob[0])
        bbox_responsible = tf.zeros_like(box_x)


        appears_mask_fn = lambda x: tf.cast(tf.reduce_max(x) > tf.constant(0, dtype = tf.float32), dtype=tf.float32)
        object_appears_mask = tf.map_fn(lambda x: tf.map_fn(appears_mask_fn, x), x)


        box2 = tf.convert_to_tensor([target_x - target_w/2, target_y - target_h/2, target_x + target_w/2, target_y + target_h/2])
        box1 = tf.convert_to_tensor([box_x - box_w/2, box_y - box_h/2, box_x + box_w/2, box_y + box_h/2])

        IOUs = IOU(box1, box2)
     




    return Loss

print(IOU(tf.constant([1,2,3,4]), tf.constant([2,3,4,5])))