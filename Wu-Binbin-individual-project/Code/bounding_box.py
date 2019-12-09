from PIL import Image, ImageDraw
import json
import os
import matplotlib.pyplot as plt


path_dir_test='/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/train-2'
all_files = os.listdir(path_dir_test)
image_files = filter(lambda x: x[-4:] == '.png', all_files)
json_files = filter(lambda x: x[-4:] == 'json', all_files)
TRANSPARENCY = .3  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)

a=[list(image_files)[713]]

for image in a:
    #image File
    path = os.path.join(path_dir_test, image)
    img = Image.open(path)
    width, height = img.size

    # creating new Image object
    img_background = Image.new("RGB", (width, height))

    #boundaryFile
    boundaryFile = os.path.join(path_dir_test, image[:-4]+'.json')
    with open(boundaryFile) as f:
        data = json.load(f)
    for bound_box in data:
        label = bound_box['category']
        shape = [(bound_box['bounding_box']['minimum']['c'],bound_box['bounding_box']['minimum']['r']),(bound_box['bounding_box']['maximum']['c'],bound_box['bounding_box']['maximum']['r'])]
        if bound_box['category']=='red blood cell':
            # create rectangle image
            img1 = ImageDraw.Draw(img, 'RGBA')
            img1.rectangle(shape, fill=(123, 12, 65) + (OPACITY,))
            # img1.rectangle(shape, fill =(123,12,65)+(OPACITY,),outline="blue")

            img1 = ImageDraw.Draw(img_background, 'RGB')
            img1.rectangle(shape, fill=(123, 12, 65))
        elif bound_box['category'] == 'ring':
            # create rectangle image
            img1 = ImageDraw.Draw(img, 'RGBA')
            img1.rectangle(shape, fill=(23, 146, 65) + (OPACITY,))
            # img1.rectangle(shape, fill =(123,12,65)+(OPACITY,),outline="blue")

            img1 = ImageDraw.Draw(img_background, 'RGB')
            img1.rectangle(shape, fill=(23, 146, 65))
        elif bound_box['category'] == 'schizont':
            # create rectangle image
            img1 = ImageDraw.Draw(img, 'RGBA')
            img1.rectangle(shape, fill=(23, 46, 165) + (OPACITY,))
            # img1.rectangle(shape, fill =(123,12,65)+(OPACITY,),outline="blue")

            img1 = ImageDraw.Draw(img_background, 'RGB')
            img1.rectangle(shape, fill=(23, 46, 165))
        elif bound_box['category'] == 'trophozoite':
            # create rectangle image
            img1 = ImageDraw.Draw(img, 'RGBA')
            img1.rectangle(shape, fill=(123, 46, 165) + (OPACITY,))
            # img1.rectangle(shape, fill =(123,12,65)+(OPACITY,),outline="blue")

            img1 = ImageDraw.Draw(img_background, 'RGB')
            img1.rectangle(shape, fill=(123, 46, 165))



plt.imshow(img)
plt.show()
plt.imshow(img_background)
plt.show()






