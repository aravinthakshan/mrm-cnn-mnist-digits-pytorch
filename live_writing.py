import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
try:
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()
except FileNotFoundError:
    print("Model file not found. Please check the path.")
    exit()

load_from_sys = True
if load_from_sys:
    try:
        hsv_value = np.load('hsv_value.npy')
    except FileNotFoundError:
        print("HSV value file not found. Please check the path.")
        exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(3, 1280)
cap.set(4, 720)
kernel = np.ones((5, 5), np.uint8)
x1, y1 = 0, 0
noise_thresh = 800
prediction = None

# Create a persistent canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def preprocess_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
    return tensor

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_sys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            # Draw on the canvas
            canvas = cv2.line(canvas, (x1, y1), (x2, y2), [255, 255, 255], 15)
        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0
    
    # Show the camera frame
    cv2.imshow('Camera', cv2.resize(frame, None, fx=0.6, fy=0.6))
    
    # Show the canvas in a separate window
    cv2.imshow('Canvas', cv2.resize(canvas, None, fx=0.6, fy=0.6))
    
    key = cv2.waitKey(1)
    if key == 27: # ESC key
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        prediction = None
    elif key == ord('p'):
        input_tensor = preprocess_canvas(canvas)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax().item()
        print(f"Prediction: {prediction}")

cv2.destroyAllWindows()
cap.release()