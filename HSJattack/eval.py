from utils import *
import sys
import random
import shutil

random.seed(685)

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

class img:
    def __init__(self, imgpath):
        self.count = 0
        rawimage = Image.open(imgpath)
        image = tf.keras.preprocessing.image.img_to_array(rawimage)
        # 299, 299, 3
        # print(np.shape(image))
        # print(type(image))
        # image = torchvision.transforms.functional.pil_to_tensor(rawimage)
        
        image = tf.cast(image, tf.float32)
        # print(type(image))
        image = image/255
        image = tf.image.resize(image, (imagesize, imagesize))
        # print(type(image))
        # Add batch dimension
        # 1, 299, 299, 3
        image = image[None, ...]

        # image = preprocess(image)
        # print(np.shape(image))

        self.imgplot = rawimage 
        
        if np.shape(image) == (1,imagesize,imagesize,1):
            image = tf.image.grayscale_to_rgb(image)
        elif np.shape(image) == (1,imagesize,imagesize,4):
            image = tf.image.grayscale_to_rgb(image)
            
        # print(np.shape(image))
        self.img = image
        # print(type(image))
        self.image_probs = get_imagenet_label(predict(image))
        self.labelindex = np.argmax(predict(image))
        origpredictions = self.image_probs[0]

        actualprediction = os.path.basename(os.path.dirname(imgpath))

        # print(self.image_probs)

        self.usable = False

        if self.image_probs[0][0] == actualprediction:
            # print(f"Imagenet prediction: {self.image_probs[0]}")
            # print(f"Label: {actualprediction}")
            self.usable = True


    def decision(self,img):
        # print(f"Count: {self.count}")
        check = get_imagenet_label(predict(img))
        if check[0][0] != self.image_probs[0][0]:
            return True
        else:
            return False


main_path = os.path.dirname(os.path.abspath("Main.ipynb")) #path of main folder
data_path = os.path.join(main_path, "imagenet_val") #path of validation data

images = {}
classfiles = os.listdir(data_path)
for clf in classfiles:
    images[clf] = os.listdir(os.path.join(data_path,clf))
# print(classfiles)

# count = 0
# total_count = 0
# for cls in images:
#     print(cls)
#     for imgfile in images[cls]:
#         total_count += 1
#         imgpath = os.path.join(data_path, cls, imgfile)

#         if img(imgpath).usable:
#             count += 1
#     print(count)

count = 0
success = 0
total_dist = 0
# save_dir = "output_images_3_iter"
save_dir = "output_images"
visited = set()
while count < 100:
    predict.counter = 0
    cls = random.choice(classfiles)
    imgfile = random.choice(images[cls])
    while imgfile in visited:
        cls = random.choice(classfiles)
        imgfile = random.choice(images[cls])
    visited.add(imgfile)
    imgpath = os.path.join(data_path, cls, imgfile)

    im = img(imgpath)
    if im.usable:
        with torch.no_grad():
            output_im, dist = hsja(im, constraint = 'l2', num_iterations=30)
        if im.decision(output_im):
            success += 1
            print("Success")
            total_dist += dist
        folder_name = imgfile.replace(".JPEG", "")
        os.makedirs(f"{save_dir}/{folder_name}")
        np.savetxt(f"{save_dir}/{folder_name}/distance.txt", np.array([dist]))
        np.savetxt(f"{save_dir}/{folder_name}/queries.txt", np.array([predict.counter]))
        display_images(output_im, save_path = f"{save_dir}/{folder_name}", img_name = imgfile)
        count += 1

np.savetxt(f"{save_dir}/success.txt", np.array([success, count, total_dist/success]))

# print(count)
# print(total_count)

# img = randomimg()
# hsja = hsja(img, constraint = 'linf', num_iterations=30)
# display_images(hsja)