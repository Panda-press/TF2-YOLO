import csv
import numpy as np
from os import walk

stage = 2

print("Starting stage {0}".format(stage))



if stage == 1:

    PATH = "D:\Dataset\OpenImage/annotations"
    SAVE_PATH = "D:\Dataset\OpenImage/annotations\Stage1"

    fieldnames = ['xPos','yPos','width','height','class']

    current_id = ""

    current_file = None

    csv_writer = None

    with open(PATH + "/oidv6-train-annotations-bbox.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for line in csv_reader:

            if (line["ImageID"] == current_id):
                xPos = (float(line["XMax"]) + float(line["XMin"])) /2
                yPos = (float(line["YMax"]) + float(line["YMin"])) /2
                width = float(line["XMax"]) - float(line["XMin"])
                height = float(line["YMax"]) - float(line["YMin"])
                bbox_class = str(line["LabelName"])

                csv_writer.writerow({'xPos': str(xPos), 'yPos': str(yPos), 'width': str(width), 'height': str(height), 'class': bbox_class})

            else:
                current_id = line["ImageID"]
                try:
                    current_file.close()
                except:
                    print("")

                current_file = open(SAVE_PATH + "/{0}.csv".format(current_id), mode="w")
                
                csv_writer = csv.DictWriter(current_file, fieldnames=fieldnames, lineterminator = '\n')

                xPos = (float(line["XMax"]) + float(line["XMin"])) /2 #were inially written the wrong way around 
                yPos = (float(line["YMax"]) + float(line["YMin"])) /2
                width = float(line["XMax"]) - float(line["XMin"])
                height = float(line["YMax"]) - float(line["YMin"])
                bbox_class = str(line["LabelName"])

                csv_writer.writeheader()
                csv_writer.writerow({'xPos': str(xPos), 'yPos': str(yPos), 'width': str(width), 'height': str(height), 'class': bbox_class})

elif stage == 2:



    PATH = "D:\Dataset\OpenImage/annotations\Stage1"

    SAVE_PATH = "D:\Dataset\OpenImage/annotations\Stage2"

    files = []

    for (dirpath, dirnames, filenames) in walk(PATH):
        files.append(filenames)

    for filename in files[0]:
        with open(PATH+"\{0}".format(filename)) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            yolo_targets = np.zeros((8, 8, 2+2+601))

            for line in csv_reader:
                xgridsquare, xgridpos = GetYoloPosition(float(line["xPos"]))
                ygridsquare, ygridpos = GetYoloPosition(float(line["yPos"]))
                yolo_targets[xgridsquare, ygridsquare, 0] = xgridpos 
                yolo_targets[xgridsquare, ygridsquare, 1] = ygridpos
                yolo_targets[xgridsquare, ygridsquare, 2] = float(line["width"])
                yolo_targets[xgridsquare, ygridsquare, 3] = float(line["height"])
                bbox_class = None

                with open("D:\Dataset\OpenImage/annotations/class-descriptions-boxable.csv") as csv_file:
                    class_reader = csv.reader(csv_file, delimiter=",")

                    for class_line in class_reader:
                        line_num = 0
                        if (class_line[0] == line["class"]):
                            bbox_class = line_num
                            break
                        line_num += 1

                yolo_targets[xgridsquare, ygridsquare, bbox_class+4] = 1
                
                np.save(SAVE_PATH + "/{0}.npy".format(filename), yolo_targets)
        

