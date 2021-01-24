import tensorflow as tf


@tf.function
def IOU(box1, box2):
    x1 = tf.reduce_max(tf.stack([box1[0], box2[0]],axis=3),axis=-1)
    y1 = tf.reduce_max(tf.stack([box1[1], box2[1]],axis=3),axis=-1)
    x2 = tf.reduce_min(tf.stack([box1[2], box2[2]],axis=3),axis=-1)
    y2 = tf.reduce_min(tf.stack([box1[3], box2[3]],axis=3),axis=-1)
    

    intersection = tf.reduce_max(tf.stack([tf.zeros_like(x1), x2 - x1], axis=3), axis=-1) * tf.reduce_max(tf.stack([tf.zeros_like(y1), y2 - y1], axis=3), axis=-1)


    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

    # tf.print("box1")
    # tf.print(box1)
    # tf.print("box2")
    # tf.print(box2)
    # tf.print("union")
    # tf.print(union)
    # tf.print("intersection")
    # tf.print(intersection)

    return tf.math.divide_no_nan(intersection, union)

def Model_Loss(bboxes, num_classes):
    @tf.function
    def Loss(Model_Output, Target_Output):
        loss = tf.constant(0, dtype=tf.float32)
        lambda_coord = tf.constant(5, dtype=tf.float32)
        lambda_noobj = tf.constant(0.5, dtype=tf.float32)
        iou_threashhold = tf.constant(0.5, dtype=tf.float32)

        # tf.print("Targets")
        # tf.print(Target_Output)
        # tf.print("Model")
        # tf.print(tf.sigmoid(Model_Output))


        box_x, box_y, box_w, box_h, objectness, class_prob = tf.split(Model_Output, [bboxes, bboxes, bboxes, bboxes, bboxes, num_classes], axis=-1)


        box_x = tf.sigmoid(box_x)   #the activation functions that the output has to go through
        box_y = tf.sigmoid(box_y)
        box_w = tf.sigmoid(box_w)
        box_h = tf.sigmoid(box_h)
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
        # print(target_boxes)
        # tf.print(target_x)

        object_appears_mask = tf.zeros_like(class_prob[0])
        bbox_responsible = tf.zeros_like(box_x)


        appears_mask_fn = lambda x: tf.cast(tf.reduce_max(x) > tf.constant(0, dtype = tf.float32), dtype=tf.float32)
        object_appears_mask = tf.map_fn(lambda x: tf.map_fn(appears_mask_fn, x), target_x)


        box1 = tf.convert_to_tensor([box_x - box_w/2, box_y - box_h/2, box_x + box_w/2, box_y + box_h/2])
        box2 = tf.convert_to_tensor([target_x - target_w/2, 
                                    target_y - target_h/2, 
                                    target_x + target_w/2, 
                                    target_y + target_h/2])

        IOUs = IOU(box1, box2)
        

        # tf.print("IOUs")
        # tf.print(IOUs)
        # print(object_appears_mask)
        IOUs += tf.tile(tf.reshape(object_appears_mask, [8,8,1]), [1,1, bboxes])

        func = lambda x: tf.map_fn(lambda w: tf.cast(tf.reduce_max(x) == w and w != 0, dtype=tf.float32), x)
        bbox_responsible = tf.map_fn(lambda x: tf.map_fn(func, x), IOUs)
        

        #tf.print("Responisibility")
        # tf.print(tf.reduce_max(IOUs))
        #tf.print(bbox_responsible)


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
        wh_squared_error = h_squared_error + w_squared_error
        wh_loss = bbox_responsible * wh_squared_error
        loss += lambda_coord * tf.reduce_sum(wh_loss)


        #objectness_delta = bbox_responsible - objectness
        #tiled_bbox_responisble = tf.tile(tf.reshape(bbox_responsible, [8,8,1]), [1,1, bboxes])
        #object_error = tf.math.pow(objectness_delta, 2)
        object_error = tf.losses.binary_crossentropy(tf.reshape(bbox_responsible, [8,8,2,1]), tf.reshape(objectness, [8,8,2,1])) 
        #print(objectness_delta)
        #print(object_error)
        #loss += tf.reduce_sum(object_error)


        not_responsible_mask = tf.ones_like(bbox_responsible) - bbox_responsible

        iou_threashhold_mask = tf.cast(iou_threashhold > IOUs,dtype=tf.float32)


        tiled_appears_mask = tf.tile(tf.reshape(object_appears_mask, [8,8,1]), [1,1,2])

        objectness_loss = tf.zeros_like(object_error)
        objectness_loss += object_error * not_responsible_mask * iou_threashhold_mask
        # tf.print("objectness_loss")
        # tf.print(objectness_loss)
        objectness_loss += object_error * bbox_responsible
        loss += tf.reduce_sum(objectness_loss)

        # tf.print("objectness")
        # tf.print(objectness)
        # tf.print("deltas")
        # tf.print(objectness_delta)
        # tf.print("responsible")
        # tf.print(bbox_responsible)
        # tf.print("not responsible")
        # tf.print(not_responsible_mask)
        # tf.print("loss")


        class_delta = class_prob - target_class
        class_squared_error = tf.math.pow(class_delta, 2)
        class_element_loss = object_appears_mask * tf.math.reduce_sum(class_squared_error, axis = -1)
        class_loss = tf.reduce_sum(class_element_loss)
        loss += class_loss
        #print(loss)

        tf.print("loss")
        tf.print(loss)

        return loss



    return Loss

