import tensorflow as tf
import numpy as np



# x = tf.random.uniform([8, 8, 3], -1, 1)
# a, b = tf.split(x, [2,1], axis=-1)
# print(x)
# print(a)



# a = tf.zeros([8, 8, 4], 1)
# b = tf.random.uniform([8, 8, 2], -1, 1)

# x = tf.constant(len(a[0,0,]))
# y = tf.constant(len(b[0,0,]))

# print(tf.constant(x))
# print(tf.constant(y))

# c = tf.tile(b, [1,1,tf.constant(len(a[0,0,]))/tf.constant(len(b[0,0,]))])

# print(c)
# print(b)
# print(a)



# def IOU(box1, box2):
#     x1 = tf.reduce_max(tf.convert_to_tensor([box1[0], box2[0]]))
#     y1 = tf.reduce_max(tf.convert_to_tensor([box1[1], box2[1]]))
#     x2 = tf.reduce_min(tf.convert_to_tensor([box1[2], box2[2]]))
#     y2 = tf.reduce_min(tf.convert_to_tensor([box1[3], box2[3]]))

#     intersection = tf.reduce_max(tf.convert_to_tensor([0, x2 - x1])) * tf.reduce_max(tf.convert_to_tensor([0, y2 - y1]))

#     union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

#     iou = intersection/union

#     return iou

# box_x = tf.ones([8, 8, 4], 1)
# box_y = tf.ones([8, 8, 4], 1)
# box_w = tf.ones([8, 8, 4], 1)
# box_h = tf.ones([8, 8, 4], 1)

# target_x = tf.ones([8, 8, 4], 1)
# target_y = tf.ones([8, 8, 4], 1)
# target_w = tf.ones([8, 8, 4], 1)
# target_h = tf.ones([8, 8, 4], 1)

# print(box_x)

# box2 = tf.convert_to_tensor([target_x - target_w/2, target_y - target_h/2, target_x + target_w/2, target_y + target_h/2])

# box1 = tf.convert_to_tensor([box_x - box_w/2, box_y - box_h/2, box_x + box_w/2, box_y + box_h/2])

# IOUs = IOU(box1, box2)

# print(IOU([0.5, 0.5, 1.5, 1.5], [0.5, 0.5, 1.5, 1.5]))

# print(IOUs)




# x = tf.random.uniform([8, 8, 2], -1, 1)

# #x = tf.zeros_like(x)

# x = x.numpy()

# x[2,5,1] = 4
# x[2,5,0] = 2

# x[7,4,0] = .1

# x = tf.convert_to_tensor(x)

# func = lambda x: tf.map_fn(lambda w: tf.cast(tf.reduce_max(x) == w, dtype=tf.float32), x)

# print(tf.reduce_max(x[2,5]))

# x = tf.map_fn(lambda x: tf.map_fn(func, x), x)



# print(x)

k = tf.ones
lambda_noobj = tf.constant(0.5, dtype=tf.float32)
lambda_noobj_mask = 