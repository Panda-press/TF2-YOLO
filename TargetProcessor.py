import csv

stage = 1

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


            


