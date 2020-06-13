import PIL
from PIL import Image
from os import walk

new_width = 512
new_height = 512

path = "D:\Dataset\OpenImage/train/train_0"
proccessed_path = "D:\Dataset\OpenImage/train/train_0_p"
files = []
for (dirpath, dirnames, filenames) in walk(path):
    files.append(filenames)

#print(files)


for image in files[0]:
    #try:
    img = Image.open(path+"\\"+image)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.convert("RGB").save(proccessed_path+"\\"+image)

    # except:
    #     print(image)

print("finised")