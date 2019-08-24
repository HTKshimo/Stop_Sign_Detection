import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():

    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02930766'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 2373

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            if pic_num > 2600:
                break
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def create_negative_images():
    pic_num = 1000
    index = 2299

    if not os.path.exists('neg'):
        os.makedirs('neg')

    while pic_num < 10000:
        try:
            print(pic_num)
            img = cv2.imread("signDatabasePublicFramesOnly/negatives/negativePics/nosign0"+str(pic_num)+".png", cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(index)+".jpg",resized_image)
            pic_num += 100
            index += 1

        except Exception as e:
            print(str(e))

def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

# store_raw_images()
# create_negative_images()
# find_uglies()
# create_pos_n_neg()

# try different numbers of stages and see which one is more accurate
# develop a way to test multiple images at the same time - such as using arrays

# this is the cascade we just made. Call what you want
stopsign_cascade = cv2.CascadeClassifier('stopsign-cascade-11stages.xml')

# test 1
cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

while 1:
    ret, img = cap.read()
    # if img == None:
    #     raise Exception("could not load image !")
    # img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # image, reject levels level weights.
    stopsigns = stopsign_cascade.detectMultiScale(img)

    # add this
    for (x,y,w,h) in stopsigns:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# test 2
# pic_count = 0
# for file_type in ['test']:
#
#     for img in os.listdir(file_type):
#         pic_count += 1
#         # print(img)
#         image = cv2.imread('test/'+img, cv2.IMREAD_GRAYSCALE)
#         stopsigns = stopsign_cascade.detectMultiScale(image)
#         count = 0
#         for (x,y,w,h) in stopsigns:
#             cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
#             count += 1
#         if count != 0:
#             cv2.imwrite('result-9stages/'+img, image)
#             print(img)
