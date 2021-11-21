# USAGE
# python static_saliency.py --image images/neymar.jpg

# import the necessary packages
import cv2
import numpy as np
import glob
import zipimport
import matplotlib.pyplot as plt

importer = zipimport.zipimporter('Assignment_2_files.zip')
code = importer.get_code('Assignment_Two')
print code

def generate_roi(salmap, thresh, alpha, beta):

    image=salmap
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]
    aspect=width/height
    threshMap = cv2.threshold(image, int(2.55*thresh), 255,cv2.THRESH_BINARY)[1]

    # find contours and get the external one
    contours, hier = cv2.findContours(threshMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi=[]
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w>15 and h>15:
            roi.append([x, y, w, h])
            # cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # print(roi)
    for r in roi:
        h_opti=round((r[2]/aspect))
        w_opti=round(r[2])
        # draw a red rectangle to visualize the bounding rect
        cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
        x1=r[0]
        y1=int(r[1]/4)

        Rs=0
        roi_final=[]
        for j in range(r[3]-h_opti):
            dr=alpha*np.sum(threshMap[r[1]+j:r[1]+j+h_opti, r[0]:r[0]+r[3]])-beta*(h_opti*r[3])
            if dr>Rs:
                Rs=dr
                roi_final=[r[0],r[1]+j,r[3],h_opti]

    return tuple(roi_final)


def mask_score_roi(mask, roi):

    temp=mask[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
    TP = np.sum(temp == 255)

    h = mask.shape[0]
    w = mask.shape[1]
    FN = np.sum(temp == 255) -TP

    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area=0
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c)>area:
            r=(x, y, w, h)
            area=cv2.contourArea(c)
    
    temp=mask[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
    FP=abs(roi[2]-r[2]) * abs(roi[3]-r[3])

    precision=TP/(TP+FN)
    recall=TP/(TP+FP)

    return precision,recall

def salience_score_roi(map, roi):
    temp=map[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
    score=np.mean(temp)
    tot=np.sum(temp>0)
    missed=tot-score

    return score,missed


# load the input image

for input1 in glob.glob("./human_maps/*.png",0):
    image_sal = cv2.imread(input1)
for input2 in glob.glob("./images/*.jpg"):
    image_orig = cv2.imread(input2)
for input3 in glob.glob("./masks/*.png",0):
    image_mask = cv2.imread(input3)

alpha=0.8
beta=0.3
thresh=50
roi_coords = generate_roi(image_sal, thresh, alpha, beta)


sals=["AIM","BMS","DGII","IKN","IMSIG","SalGAN"]
param=[[0.8,0.3],[1.0,0.1],[0.6,0.4]]



for j in range(3):
    sc=[]
    ms=[]
    th=[]
    for i in range(10,100,10):
        try:
            roi_coords = generate_roi(image_sal, i, param[j][0], param[j][1])
            score,missed = salience_score_roi(image_mask, roi_coords)
            sc.append(score)
            ms.append(missed)
            th.append(i)
        except:
            pass

    plt.figure()
    # th=[i -253 for i in th]
    plt.plot(th,sc)
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Score vs Threshold")

    plt.figure()
    # th=[i/100 for i in th]
    plt.plot(th,ms)
    plt.xlabel("threshold")
    plt.ylabel("misses")
    plt.title("Misses vs Threshold")

plt.show()

print(roi_coords)

cv2.rectangle(image_orig, (roi_coords[0], roi_coords[1]), (roi_coords[0]+roi_coords[2], roi_coords[1]+roi_coords[3]), (255, 0, 255), 2)
ROI_image=image_orig[roi_coords[1]:roi_coords[1]+roi_coords[3] , roi_coords[0]:roi_coords[0]+roi_coords[2]]
cv2.imshow("Image Salmap", image_sal)
cv2.imshow("Image Mask", image_mask)
cv2.imshow("Original", image_orig)
cv2.imshow("ROI", ROI_image)

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()