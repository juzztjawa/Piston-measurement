import tkinter as tk
import cv2
import numpy as np
import math
from tkinter import font
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter

start_x, start_y =0,150
end_x, end_y =600,380

def get_measurement(frame):

        # ret, frame = vid.read()
        # cropped_frame = frame[start_y:end_y, start_x:end_x]
        cropped_frame=frame.copy()
        image=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)
        # kernel = np.array([[0, -1, 0],
        #            [-1, 5,-1],
        #            [0, -1, 0]])

        # Apply the sharpening kernel to the image
        # sharpened_image = cv2.filter2D(image, -1, kernel)
        _, segmented_image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY )#+ cv2.THRESH_OTSU)
        edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        # cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        cv2.imshow('mask',mask)
        # for contour in contours:
        #     cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Step 5: Apply the mask to make everything inside the contour black
        # result = cv2.bitwise_and(image, image, mask=mask)
        image[mask == 255] = 0

        # cv2.imshow('Result', result)

        _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
        
        gray=edges.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            # Get the rightmost point
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            # cv2.circle(edges, leftmost, 5, (200), -1)  
            # cv2.circle(edges, rightmost, 5, (200), -1) 
            print(f"Leftmost point: {leftmost}")
            print(f"Rightmost point: {rightmost}")

        totalLength=math.dist(leftmost,rightmost)
        centre=(int((leftmost[0]+rightmost[0])/2),int((leftmost[1]+rightmost[1])/2))
        centright=(int((centre[0]+rightmost[0])/2),int((centre[1]+rightmost[1])/2))
        centleft=(int((leftmost[0]+centre[0])/2),int((leftmost[1]+centre[1])/2))

        # cv2.circle(edges, centre, 5, (200), -1)
        # cv2.circle(edges, centright, 5, (200), -1)
        # cv2.circle(edges, centleft, 5, (200), -1)
        
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=20)
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
        secondleftsmall = [-1, -1]
        secondrightsmall = [-1, -1]
        distleft_list = []
        # Create a list to store distances from centright
        distright_list = []
        # print(corners)
        for i in cornerpoints:
            # print(i)
            distleft=math.dist(leftmost,i)
            distright=math.dist(rightmost,i)
            distleft_list.append((distleft, i))
            distright_list.append((distright, i))
            # if distleft<mindistleft:
            #     mindistleft=distleft
            #     leftsmall[0],leftsmall[1]=i
            # if distright<mindistright:
            #     mindistright=distright
            #     rightsmall[0],rightsmall[1]=i
            # Sort the distances from centleft
        distleft_list.sort(key=lambda x: x[0])

        # Sort the distances from centright
        distright_list.sort(key=lambda x: x[0])

        # Get the nearest and second nearest points from centleft
        if len(distleft_list) >= 1:
            mindistleft, actualleftsmall = distleft_list[0]
        if len(distleft_list) >= 2:
            # _, secondleftsmall = distleft_list[1]
            _, leftsmall = distleft_list[1]

        # Get the nearest and second nearest points from centright
        if len(distright_list) >= 1:
            mindistright, actualrightsmall = distright_list[0]
        if len(distright_list) >= 2:
            # _, secondrightsmall = distright_list[1]
            _, rightsmall = distright_list[1]

        cv2.circle(edges, leftsmall, 3, (200), -1)
        cv2.circle(edges, rightsmall, 3, (200), -1)
        upPoint=[-1,-1]
        downPoint=[-1,-1]
        print("centre point is ",centre)
        
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
                    leftSmall2=[leftsmall[0],leftmost[1]+i]
                    break
        rightSmall2=[-1,-1]
        if rightmost[1]<rightsmall[1]:
            print("heyyyyyyyyyyyyyy")
            for i in range(segmented_image.shape[0]//2):
                # print("yesssss",rightmost[1]-i,rightsmall[0])
                # print(segmented_image[rightmost[1]-i,rightsmall[0]])
                if segmented_image[rightmost[1]-i,rightsmall[0]]==255:
                    rightSmall2=[rightsmall[0],rightmost[1]-i]
                    break
        else:
            print("niowdoihawdhawodihaodwih")
            for i in range(segmented_image.shape[0]//2):
                # print("yesssss",rightmost[1]-i,rightsmall[0])
                # print(segmented_image[rightmost[1]-i,rightsmall[0]])
                if segmented_image[rightmost[1]-i,rightsmall[0]]==255:
                    rightSmall2=[rightsmall[0],rightmost[1]+i]
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
        
        totalLength=round((totalLength*0.6),2)
        centreWidth=round((centreWidth*0.6),2)
        leftLength=round((leftLength*0.6),2)
        rightLength=round((rightLength*0.6),2)
        leftWidth=round((leftWidth*0.6),2)
        rightWidth=round((rightWidth*0.6),2)
        print("Total length of the entire part = ",totalLength)
        print("Length of the centre part = ",totalLength-leftLength-rightLength)
        print("Width of the centre part = ",centreWidth)
        print("Length of the left part = ",leftLength)
        print("Length of the right part = ",rightLength)
        print("Width of the left part = ",leftWidth)
        print("Width of the right part = ",rightWidth)
        # cv2.circle(edges, leftsmall, 3, (200), -1)
        # cv2.circle(edges, rightsmall, 3, (200), -1)
        cv2.imshow('edge',edges)




 
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    frame=frame[start_y:end_y, start_x:end_x]
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('s'): 
        get_measurement(frame)
        # break

    elif cv2.waitKey(1) & 0xFF == ord('q'): 
        # get_measurement(frame)
        break
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

