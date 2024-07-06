import cv2
import numpy as np
import math
# Read the image

origimage = cv2.imread('dataset/8.jpeg')
origimage2=origimage.copy()
image=cv2.cvtColor(origimage,cv2.COLOR_BGR2GRAY)
# image=cv2.resize(image,(1000,1000))
# kernel = np.ones((5,5),np.uint8)
# dilated_edges = cv2.dilate(image, kernel, iterations=1)
# Apply Otsu's thresholding
# image=cv2.GaussianBlur(image,(5,5),5)
_, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, segmented_image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
cv2.imshow('edges',edges)
# cv2.imshow('original',image)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('contour',contours)
mask = np.zeros_like(image)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
# cv2.imshow('mask',mask)
for contour in contours:
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Step 5: Apply the mask to make everything inside the contour black
result = cv2.bitwise_and(image, image, mask=mask)
image[mask == 255] = 0

# cv2.imshow('Result', image)

_, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
gray=edges.copy()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    # Get the rightmost point
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    cv2.circle(edges, leftmost, 5, (200), -1)  
    cv2.circle(edges, rightmost, 5, (200), -1) 
    print(f"Leftmost point: {leftmost}")
    print(f"Rightmost point: {rightmost}")

totalLength=round(math.dist(leftmost,rightmost),2)
centre=(int((leftmost[0]+rightmost[0])/2),int((leftmost[1]+rightmost[1])/2))
centright=(int((centre[0]+rightmost[0])/2),int((centre[1]+rightmost[1])/2))
centleft=(int((leftmost[0]+centre[0])/2),int((leftmost[1]+centre[1])/2))

cv2.circle(edges, centre, 5, (200), -1)
cv2.circle(edges, centright, 5, (200), -1)
cv2.circle(edges, centleft, 5, (200), -1)
# print(totalLength,centre,centright,centleft)
# cv2.imshow('edges',edges)

#############HARRIS CORNER IMPLEMENTATION
# gray=edges.copy()
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.1)
# print(dst)
# # Step 3: Dilate the corner image to enhance the corner points
# dst = cv2.dilate(dst, None)
# cv2.imshow('dst',dst)
# # Threshold for an optimal value, it may vary depending on the image
# edges[dst > 0.01 * dst.max()] = [200]  # Mark corners in red

# # Display the result
# cv2.imshow('Harris Corners', edges)

# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#                            param1=50, param2=30, minRadius=0, maxRadius=1000)
# print(circles)
#####################################

#####Shi-Tomasi Implementation
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.4, minDistance=20)
corners = np.intp(corners)

# Step 3: Draw circles on the detected corners
cornerpoints=[]
for corner in corners:
    x, y = corner.ravel()
    cornerpoints.append((x,y))
    # cv2.circle(edges, (x, y), 3, (200), -1)

mindistleft=10000000000
mindistright=1000000000
leftsmall=[-1,-1]
rightsmall=[-1,-1]
# print(corners)
for i in cornerpoints:
    # print(i)
    distleft=math.dist(centleft,i)
    distright=math.dist(centright,i)
    if distleft<mindistleft:
        mindistleft=distleft
        leftsmall[0],leftsmall[1]=i
    if distright<mindistright:
        mindistright=distright
        rightsmall[0],rightsmall[1]=i

up,down=0,0
upPoint=[-1,-1]
downPoint=[-1,-1]
print("centre point is ",centre)
# for i in range(segmented_image.shape[1]):
#     for j in range(segmented_image.shape[0]):
#         if segmented_image[j,i]==0:
#             print(i,j)
#             cv2.circle(segmented_image, (i,j), 3, (200), -1)
for i in range(segmented_image.shape[0]//2):
    # cv2.circle(segmented_image,(centre[0],centre[1]+i), 3, (200), -1)
    if segmented_image[centre[1]+i,centre[0]]==255:
        upPoint=[centre[0],centre[1]+i]
        break
for i in range(segmented_image.shape[0]//2):
    # cv2.circle(segmented_image,(centre[0],centre[1]+i), 3, (200), -1)
    if segmented_image[centre[1]-i,centre[0]]==255:
        downPoint=[centre[0],centre[1]-i]
        break

# cv2.circle(segmented_image,downPoint, 3, (200), -1)
# cv2.circle(segmented_image,upPoint, 3, (200), -1)
leftSmall2=[-1,-1]
if leftmost[1]<leftsmall[1]:
    for i in range(segmented_image.shape[0]//2):
        # print("yesssss",leftmost[1]-i,leftsmall[0])
        # print(segmented_image[leftmost[1]-i,leftsmall[0]])
        if segmented_image[leftmost[1]-i,leftsmall[0]]==255:
            leftSmall2=[leftsmall[0],leftmost[1]-i]
            break
else:
    for i in range(segmented_image.shape[0]//2):
        # print("yesssss",leftmost[1]-i,leftsmall[0])
        # print(segmented_image[leftmost[1]-i,leftsmall[0]])
        if segmented_image[leftmost[1]-i,leftsmall[0]]==255:
            leftSmall2=[leftsmall[0],leftmost[1]-i]
            break
rightSmall2=[-1,-1]
if rightmost[1]<rightsmall[1]:
    for i in range(segmented_image.shape[0]//2):
        # print("yesssss",rightmost[1]-i,rightsmall[0])
        # print(segmented_image[rightmost[1]-i,rightsmall[0]])
        if segmented_image[rightmost[1]-i,rightsmall[0]]==255:
            rightSmall2=[rightsmall[0],rightmost[1]-i]
            break
else:
    for i in range(segmented_image.shape[0]//2):
        # print("yesssss",rightmost[1]-i,rightsmall[0])
        # print(segmented_image[rightmost[1]-i,rightsmall[0]])
        if segmented_image[rightmost[1]-i,rightsmall[0]]==255:
            rightSmall2=[rightsmall[0],rightmost[1]-i]
            break
print("left corner points",leftsmall,leftSmall2)
print("right corner points",rightsmall,rightSmall2)
cv2.circle(edges, leftSmall2, 3, (200), -1)
cv2.circle(edges, rightSmall2, 3, (200), -1)
centreWidth=math.dist(downPoint,upPoint)
leftLength=abs(leftsmall[0]-leftmost[0])
leftWidth=math.dist(leftsmall,leftSmall2)
rightWidth=math.dist(rightsmall,rightSmall2)
rightLength=abs(rightsmall[0]-rightmost[0])

totalLength=round((totalLength*0.18),2)
centreWidth=round((centreWidth*0.18),2)
leftLength=round((leftLength*0.18),2)
rightLength=round((rightLength*0.18),2)
leftWidth=round((leftWidth*0.18),2)
rightWidth=round((rightWidth*0.18),2)

print("Total length of the entire part = ",totalLength)
print("Length of the centre part = ",totalLength-leftLength-rightLength)
print("Width of the centre part = ",centreWidth)
print("Length of the left part = ",leftLength)
print("Length of the right part = ",rightLength)
print("Width of the left part = ",leftWidth)
print("Width of the right part = ",rightWidth)
cv2.circle(edges, leftsmall, 3, (200), -1)
cv2.circle(edges, rightsmall, 3, (200), -1)
# Display the result


cv2.line(origimage,(leftsmall[0]-70,leftsmall[1]),(leftSmall2[0]-70,leftSmall2[1]+5),(255,100,100),2)
cv2.putText(origimage,str(float(leftWidth)),(leftSmall2[0]-110,leftSmall2[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,30,150),2)
cv2.line(origimage,(leftmost[0],leftmost[1]-40),(rightmost[0],rightmost[1]-40),(255,0,0),2)
cv2.putText(origimage,str(totalLength),(leftmost[0],leftmost[1]-60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
cv2.line(origimage,(leftmost[0],leftmost[1]+40),(leftsmall[0],leftsmall[1]+20),(0,0,0),2)
cv2.putText(origimage,str(leftLength),(leftmost[0],leftmost[1]+70),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
cv2.line(origimage,(rightmost[0],rightmost[1]+35),(rightsmall[0],rightsmall[1]+20),(0,255,255),2)
cv2.putText(origimage,str(rightLength),(rightmost[0],rightmost[1]+70),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
cv2.line(origimage,(rightsmall[0]+45,rightsmall[1]),(rightSmall2[0]+45,rightSmall2[1]),(255,0,255),2)
cv2.putText(origimage,str(float(rightWidth)),(rightSmall2[0]+50,rightSmall2[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
cv2.line(origimage,(rightsmall[0]+80,rightsmall[1]),(rightSmall2[0]+80,rightSmall2[1]-8),(255,255,255),2)
cv2.putText(origimage,str(float(centreWidth)),(rightSmall2[0]+85,rightSmall2[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

# cv2.imshow('Shi-Tomasi Corners', edges)
point=(340,450)
# te="yeah new\n line\n must work"
# cv2.putText(origimage,str(te),point,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
cv2.imshow('orig image',origimage)

# cv2.imshow('segment',segmented_image)
# Find connected components

# cv2.circle(image, centre, 5, (200), -1)
# cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
