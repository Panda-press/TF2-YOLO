import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tensorflow as tf

size = 512

class Bbox:
    def __init__(self, bottom_left, width, height, obj_class):
        self.bottom_left = bottom_left
        self.width = width
        self.height = height
        self.obj_class = obj_class

    @staticmethod
    def Plot(box_data):
        if (isinstance(box_data, Bbox)):
            rect = patches.Rectangle((box_data.bottom_left[0] * size, box_data.bottom_left[1] * size), box_data.width * size, box_data.height * size, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(2, 20, str(box_data.obj_class), color='r')
        else:
            for box in box_data:
                Bbox.Plot(box)



def ConvertModelOutput(Model_Output, classes):
    try:
        box_x, box_y, box_w, box_h, objectness, class_prob = tf.split(Model_Output[0], [2, 2, 2, 2, 2, classes], axis=-1)
    except:
        box_x, box_y, box_w, box_h, objectness, class_prob = tf.split(Model_Output[0], [2, 2, 2, 2, 2, len(classes)], axis=-1)        

    box_x = tf.sigmoid(box_x)
    box_y = tf.sigmoid(box_y)
    box_w = tf.sigmoid(box_w)
    box_h = tf.sigmoid(box_h)
    objectness = tf.sigmoid(objectness)
    tf.print("objectness")
    tf.print(objectness[0,2])
    class_prob = tf.sigmoid(class_prob)

    bboxes = []

    for x in range (0,8):
        for y in range (0,8):
            if (objectness[x,y,0] > 0.7 or objectness[x,y,1] > 0.7):
                bbox_to_use = int(objectness[x,y,0] < objectness[x,y,1])
                tf.print(box_w[x,y,bbox_to_use])
                #tf.print(objectness)
                
                this_box_x = box_x[x,y,bbox_to_use]
                this_box_y = box_y[x,y,bbox_to_use]
                this_box_w = box_w[x,y,bbox_to_use]
                this_box_h = box_h[x,y,bbox_to_use]

                tf.print("success")

                class_to_use = tf.math.argmax(class_prob)

                bboxes.append( Bbox(( (this_box_x + x)/8 - this_box_w/2 , (this_box_y + y)/8 - this_box_h/2 ), 
                                    this_box_w, 
                                    this_box_h, 
                                    class_to_use) )

                tf.print ((this_box_x - this_box_w/2 + x)/8)
                
                tf.print ((this_box_x - this_box_w + x)/8)

    return bboxes
                

