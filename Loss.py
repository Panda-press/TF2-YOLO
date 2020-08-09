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

def Model_Loss(bboxes, num_classes):
    @tf.function
    def Loss(Model_Output, Target_Output):
        loss = tf.constant(0, dtype=tf.float32)
        lambda_coord = tf.constant(5, dtype=tf.float32)
        lambda_noobj = tf.constant(0.5, dtype=tf.float32)


        box_x, box_y, box_w, box_h, objectness, class_prob = tf.split(Model_Output, [bboxes, bboxes, bboxes, bboxes, bboxes, num_classes], axis=-1)

        box_x = tf.sigmoid(box_x)
        box_y = tf.sigmoid(box_y)
        box_w = tf.exp(box_w)
        box_h = tf.exp(box_h)
        objectness = tf.sigmoid(objectness)
        class_prob = tf.sigmoid(class_prob)


        target_boxes, target_class = tf.split(Target_Output, [4, num_classes], axis=-1)
        
        # target_boxes = tf.tile(target_boxes, [1,1, (tf.constant(4, dtype=tf.int32) * tf.shape(box_w)[2])//tf.shape(target_boxes)[2]])
        # #target_boxes = tf.tile(target_boxes, [1,1,tf.constant(len(box_w[0,0,]))/tf.constant(len(target_boxes[0,0,])/tf.constant(4))])
        # print(target_boxes)
        # target_x, target_y, target_w, target_h = tf.split(target_boxes, [bboxes, bboxes, bboxes, bboxes], axis=-1)
        # print(target_x)
        # print(box_x)

        #fix target splitting - done
        #it's tiled x,y,w,h to x,y,w,h,x,y,w,h
        #and is then split x,y w,h x,y w,h
        #done

        target_x, target_y, target_w, target_h = tf.split(target_boxes, [1,1,1,1], axis=-1)
        target_x = tf.tile(target_x, [1,1, bboxes])
        target_y = tf.tile(target_y, [1,1, bboxes])
        target_w = tf.tile(target_w, [1,1, bboxes])
        target_h = tf.tile(target_h, [1,1, bboxes])
        print(target_boxes)
        print(target_x)

        object_appears_mask = tf.zeros_like(class_prob[0])
        bbox_responsible = tf.zeros_like(box_x)


        appears_mask_fn = lambda x: tf.cast(tf.reduce_max(x) > tf.constant(0, dtype = tf.float32), dtype=tf.float32)
        object_appears_mask = tf.map_fn(lambda x: tf.map_fn(appears_mask_fn, x), target_x)


        box2 = tf.convert_to_tensor([target_x - target_w/2, target_y - target_h/2, target_x + target_w/2, target_y + target_h/2])
        box1 = tf.convert_to_tensor([box_x - box_w/2, box_y - box_h/2, box_x + box_w/2, box_y + box_h/2])
        IOUs = IOU(box1, box2)
        print(IOUs)

        func = lambda x: tf.map_fn(lambda w: tf.cast(tf.reduce_max(x) == w, dtype=tf.float32), x)
        bbox_responsible = tf.map_fn(lambda x: tf.map_fn(func, x), IOUs)


        x_delta = box_x - target_x
        x_squared_error = tf.math.pow(x_delta, 2)
        y_delta = box_y - target_y
        y_squared_error = tf.math.pow(y_delta, 2)
        pos_squared_error = x_squared_error + y_squared_error
        pos_loss = bbox_responsible * pos_squared_error
        loss += lambda_coord * tf.reduce_sum(pos_loss)


        w_delta = tf.math.sqrt(box_w) - tf.math.sqrt(target_w)
        w_squared_error = tf.math.pow(w_delta, 2)
        h_delta = tf.math.sqrt(box_h) - tf.math.sqrt(target_h)
        h_squared_error = tf.math.pow(h_delta, 2)
        wh_squared_error = x_squared_error + y_squared_error
        wh_loss = bbox_responsible * wh_squared_error
        loss += lambda_coord * tf.reduce_sum(wh_loss)


        objectness_delta = objectness - bbox_responsible
        object_squared_error = tf.math.pow(objectness_delta, 2)

        lambda_noobj_mask = bbox_responsible - tf.ones_like(bbox_responsible)
        lambda_noobj_mask = lambda_noobj_mask * lambda_noobj
        lambda_noobj_mask = lambda_noobj_mask + tf.ones_like(bbox_responsible)

        objectness_loss = object_squared_error * lambda_noobj_mask
        loss += tf.reduce_sum(objectness_loss)


        class_delta = class_prob - target_class
        class_squared_error = tf.math.pow(class_delta, 2)
        class_element_loss = object_appears_mask * tf.math.reduce_sum(class_squared_error, axis = -1)
        print(class_element_loss)
        class_loss = tf.reduce_sum(class_element_loss)
        print(class_loss)
        loss += class_loss
        print(loss)

        return loss



    return Loss

