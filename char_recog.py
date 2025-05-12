import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras
from rotate_and_recrop_lp import auto_rotate_and_crop_lp
load_model=keras.models.load_model

def display(img_, title=''):
    if img_ is None or img_.size == 0:
        raise ValueError("Input image is empty or not loaded correctly.")
    # img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    # # fig = plt.figure(figsize=(10,6))
    # # ax = plt.subplot(111)
    # # ax.imshow(img)
    # # plt.axis('off')
    # # plt.title(title)
    # # plt.show()

def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = img.copy()

    x_cntr_list = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)

            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            # cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            # plt.imshow(ii, cmap='gray')

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)
    plt.show()
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res

def is_two_line_plate(image):
    h, w, _ = image.shape
    aspect_ratio = w / h
    return aspect_ratio < 2.0

def segment_characters_two_line(image):
    img_lp = cv2.resize(image, (200, 200),cv2.INTER_LINEAR)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[197:200,:] = 255
    img_binary_lp[:,197:200] = 255

    upper_half = img_binary_lp[0:100, :]
    lower_half = img_binary_lp[100:200, :]

    dimensions_upper = [upper_half.shape[0]/6,
                        upper_half.shape[0]/2,
                        upper_half.shape[1]/10,
                        2*upper_half.shape[1]/3]
    
    dimensions_lower = [lower_half.shape[0]/6, 
                        lower_half.shape[0]/2,
                        lower_half.shape[1]/10,
                        2*lower_half.shape[1]/3]

    char_list_upper = find_contours(dimensions_upper, upper_half)
    char_list_lower = find_contours(dimensions_lower, lower_half)
    
    return np.concatenate((char_list_upper, char_list_lower), axis=0) if char_list_upper.size > 0 and char_list_lower.size > 0 else []

def segment_characters(image):
    if is_two_line_plate(image):
        return segment_characters_two_line(image)
    else:
        img_lp = cv2.resize(image, (333, 75), cv2.INTER_LINEAR)
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        dimensions = [LP_WIDTH/6,
                      LP_WIDTH/2,
                      LP_HEIGHT/10,
                      2*LP_HEIGHT/3]

        char_list = find_contours(dimensions, img_binary_lp)

        return char_list

def lp_char_recog(plate_image):
    char = segment_characters(plate_image)
    
    if len(char) == 0:
        return "No characters detected"

    model_path = 'models/model_biensoxe (1).h5'
    model = load_model(model_path)

    def fix_dimension(img):
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for ch in char:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_NEAREST)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_ = np.argmax(model.predict(img), axis=-1)[0]
        character = dic[y_]
        output.append(character)

    plate_number = ''.join(output)
    return plate_number


def show_lp_char_recog(plate_image):
    segmented_chars = segment_characters(plate_image)
    
    if len(segmented_chars) == 0:
        print("No characters detected. Check if the image is clear and properly cropped.")
        return

    model_path = 'models/lp_char_recog_model_v2.h5'
    model = load_model(model_path)

    def fix_dimension(img): 
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    dic = {}
    characters = '0123456789ABCDEFGHKLMNPQRSTUVXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    plt.figure(figsize=(10, 2))
    for i, char_img in enumerate(segmented_chars):
        img_ = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_NEAREST)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_ = np.argmax(model.predict(img), axis=-1)[0]
        character = dic[y_]

        plt.subplot(1, len(segmented_chars), i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.axis('off')
        plt.title(character, fontsize=12)
    plt.suptitle(" ")
    plt.show()
def recognize_character(plate_image):

    plate_type = "Two-line" if is_two_line_plate(plate_image) else "One-line"
    print(f"Detected plate type: {plate_type}")
    result = lp_char_recog(plate_image)
    print("Recognized License Plate:", result)
    show_lp_char_recog(plate_image)
    return result

if __name__=="__main__":
    plate_image = cv2.imread("dataset/cropped/carlong_0934_0.jpg")

    angle, binary_img, rotated_lp = auto_rotate_and_crop_lp(plate_image)

    recognize_character(rotated_lp)