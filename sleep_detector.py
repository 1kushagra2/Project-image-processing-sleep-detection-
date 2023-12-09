
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torchvision import datasets, transforms
import cv2
import mediapipe as mp
from PIL import Image
from torch.autograd import Variable
import time
awake_count = 0
sleep_count = 0

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to match LeNet input size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


model = LeNet()

model.load_state_dict(torch.load("./Final_model.pth"))

cap = cv2.VideoCapture(0) 

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    frame = cv2.resize(frame, (32, 32))  # Resize frame to match LeNet input size
    frame = transforms.ToTensor()(frame)  # Convert frame to tensor
    frame = transforms.Normalize((0.5,), (0.5,))(frame)  # Normalize frame
    return frame.unsqueeze(0)  # Add batch dimension

# Test the model on real-time video

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
folder_name = "awake"
accurate_right_eye = [7,25,153,154,155,157,158,159,163,161,173,243,246]
# accurate_left_eye = [256,252,523,254,249,255,263,339,362,374,373,387,390,382,398,384,385,386,388,382,424,446,466]
accurate_left = [523,249,263,362,374,373,387,390,382,398,384,385,386,388,382,466]
full_left_eye = [256,252,523,254,249,255,263,339,362,374,373,387,390,382,398,384,385,386,388,382,414,446,466]
full_right_eye = [7,22,23,24,25,26,27,28,29,30,56,112,113,130,133,190,153,154,155,157,158,159,163,161,173,243,246]



def run():
  global last
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.9,
      min_tracking_confidence=0.9) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        # print(len(results.multi_face_landmarks))
        for face_landmarks in results.multi_face_landmarks:
          left_top_x = 1000000000
          left_top_y = 1000000000
          right_bottom_x = 0
          right_bottom_y = 0
          margin = 5
          for landmark_id, landmark in enumerate(face_landmarks.landmark):
              if landmark_id in full_left_eye:
                landmark_x = int(landmark.x * image.shape[1])
                landmark_y = int(landmark.y * image.shape[0])
                landmark_z = landmark.z
                if left_top_x > landmark_x:
                  left_top_x = landmark_x
                if left_top_y >landmark_y:
                  left_top_y = landmark_y
                if right_bottom_x < landmark_x:
                  right_bottom_x = landmark_x
                if right_bottom_y <landmark_y:
                  right_bottom_y = landmark_y

          if right_bottom_y - left_top_y != 20:
            right_bottom_y = left_top_y + 20 
          if right_bottom_x - left_top_x != 40:
            right_bottom_x = left_top_x + 40 
          cropped_image_left_eye = image[left_top_y-margin:right_bottom_y-margin, left_top_x-margin:right_bottom_x-margin]
          gray_cropped_image_left_eye = cv2.cvtColor(cropped_image_left_eye, cv2.COLOR_BGR2GRAY)

          left_top_x = 1000000000
          left_top_y = 1000000000
          right_bottom_x = 0
          right_bottom_y = 0

          margin = 0
          for landmark_id, landmark in enumerate(face_landmarks.landmark):
              
              if landmark_id in full_right_eye:
                landmark_x = int(landmark.x * image.shape[1])
                landmark_y = int(landmark.y * image.shape[0])
                landmark_z = landmark.z
                
                if left_top_x > landmark_x:
                  left_top_x = landmark_x
                
                if left_top_y >landmark_y:
                  left_top_y = landmark_y
                
                if right_bottom_x < landmark_x:
                  right_bottom_x = landmark_x

                if right_bottom_y <landmark_y:
                  right_bottom_y = landmark_y
                
          if right_bottom_y - left_top_y != 20:
            right_bottom_y = left_top_y + 20 
          if right_bottom_x - left_top_x != 40:
            right_bottom_x = left_top_x + 40 
          cropped_image_right_eye = image[left_top_y-margin:right_bottom_y+margin, left_top_x-margin:right_bottom_x+margin]
          gray_cropped_image_right_eye = cv2.cvtColor(cropped_image_right_eye, cv2.COLOR_BGR2GRAY)


        processed_frame = preprocess_frame(gray_cropped_image_right_eye)
        outputs = model(processed_frame)
        _, predicted = torch.max(outputs.data, 1)

        Right_prediction_text = "Awake" if predicted.item() == 0 else "Sleep"

        processed_frame = preprocess_frame(gray_cropped_image_left_eye)
        outputs = model(processed_frame)
        _, predicted = torch.max(outputs.data, 1)

        Left_prediction_text = "Awake" if predicted.item() == 0 else "Sleep"
        print(Right_prediction_text,Left_prediction_text)
        cv2.putText(image, Right_prediction_text +" "+Left_prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Prediction', image)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    
    

run()
cap.release()
cv2.destroyAllWindows()