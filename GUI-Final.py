import tkinter as tk
import cv2
import numpy as np
import math
from tkinter import font
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter


start_x, start_y =0,150
end_x, end_y =600,380
  
class Webcam:
    def __init__(self,window,window_title,video_source=0,fontObj=None):
        self.window=window
        self.window.title(window_title)
        current_font=font.Font(size=16)

        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)

        self.canvas_webcam = tk.Canvas(window, width = abs(end_x - start_x), height = abs(end_y - start_y))
        self.canvas_webcam.grid(row=0, column=0, padx=5, pady=10)

        self.canvas_processed = tk.Canvas(window, width = abs(end_x - start_x), height = abs(end_y - start_y))
        self.canvas_processed.grid(row=2, column=0, padx=10, pady=10)

        self.label_modified=None
        self.listbox=None

        self.get_measures = tk.Button(window, text="Get Measurements", width=30, command=self.get_measurement, font=fontObj)
        self.get_measures.grid(row=1, column=0, padx=5, pady=10)

        self.update()
        self.window.mainloop()
    
    def get_measurement(self):

        ret, frame = self.vid.read()
        cropped_frame = frame[start_y:end_y, start_x:end_x]
        origimage=cropped_frame.copy()
        image=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

        # Apply the sharpening kernel to the image
        # sharpened_image = cv2.filter2D(image, -1, kernel)
        _, segmented_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY )#+ cv2.THRESH_OTSU)
        edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('contour',contours)
        mask = np.zeros_like(image)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        # cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        # cv2.imshow('mask',mask)
        # for contour in contours:
        #     cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

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
                print("yesssss",rightmost[1]-i,rightsmall[0])
                # print(segmented_image[rightmost[1]-i,rightsmall[0]])
                if segmented_image[rightmost[1]-i,rightsmall[0]]==255:
                    rightSmall2=[rightsmall[0],rightmost[1]+i]
                    break
        print("left corner points",leftsmall,leftSmall2)
        print("right corner points",rightsmall,rightSmall2)
        cv2.circle(edges, leftSmall2, 3, (200), -1)
        # cv2.circle(edges, rightSmall2, 3, (200), -1)
        centreWidth=math.dist(downPoint,upPoint)
        leftLength=abs(leftsmall[0]-leftmost[0])
        leftWidth=math.dist(leftsmall,leftSmall2)
        rightWidth=math.dist(rightsmall,rightSmall2)
        rightLength=abs(rightsmall[0]-rightmost[0])
        
        totalLength=round((totalLength*0.8),2)
        centreWidth=round((centreWidth*0.63),2)
        leftLength=round((leftLength*0.4),2)
        rightLength=round((rightLength*0.4),2)
        leftWidth=round((leftWidth*0.63),2)
        rightWidth=round((rightWidth*0.63),2)
        print("Total length of the entire part = ",totalLength)
        print("Length of the centre part = ",totalLength-leftLength-rightLength)
        print("Width of the centre part = ",centreWidth)
        print("Length of the left part = ",leftLength)
        print("Length of the right part = ",rightLength)
        print("Width of the left part = ",leftWidth)
        print("Width of the right part = ",rightWidth)
        # cv2.circle(edges, leftsmall, 3, (200), -1)
        # cv2.circle(edges, rightsmall, 3, (200), -1)

        cv2.line(origimage,(leftsmall[0]-50,leftsmall[1]),(leftSmall2[0]-50,leftSmall2[1]),(255,100,100),2)
        cv2.putText(origimage,str(float(leftWidth)),(leftSmall2[0]-100,leftSmall2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,30,150),2)
        cv2.line(origimage,(leftmost[0],leftmost[1]-40),(rightmost[0],rightmost[1]-40),(255,0,0),2)
        cv2.putText(origimage,str(totalLength),(leftmost[0],leftmost[1]-60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        cv2.line(origimage,(leftmost[0],leftmost[1]+20),(leftsmall[0],leftmost[1]+20),(0,0,0),2)
        cv2.putText(origimage,str(leftLength),(leftmost[0],leftmost[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        cv2.line(origimage,(rightmost[0],rightmost[1]+20),(rightsmall[0],rightmost[1]+20),(0,255,255),2)
        cv2.putText(origimage,str(rightLength),(rightsmall[0]-20,rightmost[1]+50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        # cv2.line(origimage,(rightsmall[0]+45,rightsmall[1]),(rightSmall2[0]+45,rightSmall2[1]),(255,0,255),2)
        # cv2.putText(origimage,str(float(rightWidth)),(rightSmall2[0]+50,rightSmall2[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        cv2.line(origimage,(rightsmall[0]+30,rightsmall[1]),(rightSmall2[0]+30,rightSmall2[1]),(255,0,255),2)
        cv2.putText(origimage,str(float(centreWidth)),(rightSmall2[0]+35,rightSmall2[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

        self.display_frame(self.canvas_processed, origimage)

    def display_frame(self, canvas, frame):
        # frame=cv2.line(frame, (50,50),(100,100), (0,255,0),1)
        # frame = frame[start_y:end_y, start_x:end_x]
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep a reference to avoid garbage collection
    
    def update(self):
        ret, frame = self.vid.read()
        frame = frame[start_y:end_y, start_x:end_x]
         
        if ret:
            # frame=cv2.line(frame, (50,50),(100,100), (0,255,0),1)
            self.display_frame(self.canvas_webcam, frame)
        self.window.after(10, self.update)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


root = tk.Tk()
current_style = ttk.Style()
# current_style.configure("TLabel", font=("TkDefaultFont", 20))
fontObj = font.Font(size=20)
# root.protocol("WM_DELETE_WINDOW", on_closing)
app = Webcam(root, "ImageBee", fontObj=fontObj)