import cv2
import numpy as np
import math
# Read the image

image = cv2.imread('dataset/8.jpeg', 0)  # Read as grayscale

# image=cv2.resize(image,(1000,1000))
# kernel = np.ones((5,5),np.uint8)
# dilated_edges = cv2.dilate(image, kernel, iterations=1)
# Apply Otsu's thresholding
# image=cv2.GaussianBlur(image,(5,5),5)
_, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, segmented_image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
# cv2.imshow('edges',edges)
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

cv2.imshow('Result', image)

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
centre=(int((leftmost[0]+rightmost[0])/2),int((leftmost[1]+rightmost[1])/2))
centright=(int((centre[0]+rightmost[0])/2),int((centre[1]+rightmost[1])/2))
centleft=(int((leftmost[0]+centre[0])/2),int((leftmost[1]+centre[1])/2))

print("distanceeeeee",math.dist(centre,centright))
cv2.circle(edges, centre, 5, (200), -1)
cv2.circle(edges, centright, 5, (200), -1)
cv2.circle(edges, centleft, 5, (200), -1)
print(centre,centright,centleft)
# cv2.imshow('edges',edges)
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

print(leftsmall,rightsmall)
cv2.circle(edges, leftsmall, 3, (200), -1)
cv2.circle(edges, rightsmall, 3, (200), -1)
# Display the result
cv2.imshow('Shi-Tomasi Corners', edges)
# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

# largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
# stats is a matrix where each row contains statistics for each connected component:
# [connected component label, leftmost pixel x-coordinate, topmost pixel y-coordinate,
# width, height, area (in pixels)]
# The first row contains statistics for the background, which we usually ignore.
# print(stats)
# Print the size of each connected component (excluding the background)
# for i in range(1, num_labels):
#     area = stats[i, cv2.CC_STAT_AREA]
#     print(f'Connected Component {i}: Size = {area} pixels')


for i in range(num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    centroid = centroids[i]
    print(f'Component {i}:')
    print(f'  Position: ({x}, {y})')
    print(f'  Size: {width}x{height}')
    print(f'  Area: {area} pixels')
    print(f'  Centroid: ({centroid[0]}, {centroid[1]})')

# Optionally, display the labeled image
# Normalize the labels image to display it with different colors
image = cv2.circle(image, (x,y), radius=2, color=(255), thickness=-1)

labels = labels.astype(np.uint8)
labels = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow('Labeled Image', labels)


# # Display the original and segmented images
# cv2.imshow('Original Image', image)
# cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
