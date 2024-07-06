import cv2
import numpy as np

# Read the image
image = cv2.imread('dataset/2.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edges',edges)
# Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Draw the detected lines on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        # Filter for horizontal lines (theta near 0 or pi)
        if (theta < np.pi / 180 * 10) or (theta > np.pi / 180 * 170):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Horizontal Lines Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
