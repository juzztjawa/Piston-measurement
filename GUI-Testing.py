import tkinter as tk
import cv2
import numpy as np
import math
from tkinter import font
from tkinter import ttk
from PIL import Image, ImageTk

start_x, start_y =300,300
end_x, end_y =500,500

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
        image=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)
        _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
        self.display_frame(self.canvas_processed, edges)

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