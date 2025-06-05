Plant Leaf Detection and Disease Recognition System

This repository contains a real-time plant leaf detection and disease recognition system. The project uses a combination of YOLOv5 for leaf detection and a TensorFlow model for disease classification.

Features
Home Page: An introductory page that describes the system's functionality.
About Page: Details about the dataset and the system's goals.
Disease Recognition Page: Allows users to upload an image of a plant leaf for disease prediction.
Real-Time Detection Page: Activates the webcam to detect plant leaves and predict diseases in real-time.


Interfaces

1. Home Page
Provides an overview of the system's purpose and functionality.
  ![Screenshot 2024-12-14 193327](https://github.com/user-attachments/assets/d7e27b1f-6f8f-4ee5-a951-c8b833bb3745)

![Screenshot 2024-12-14 193415](https://github.com/user-attachments/assets/92ef30f8-c5fe-4b24-8ac8-af21176c047f)

Includes a description of how to use the application.

2. About Page

![Screenshot 2024-12-14 193431](https://github.com/user-attachments/assets/3ac2ea61-3a8c-406b-98fb-eaeaeadfb287)

Describes the dataset and its structure:

Training images: 70,295

Validation images: 17,572

Test images: 33

Explains the significance of detecting plant diseases.

3. Disease Recognition Page

![Screenshot 2024-12-14 193510](https://github.com/user-attachments/assets/76b71f47-7bce-45b9-a93b-fda84751e849)

![Screenshot 2024-12-14 193535](https://github.com/user-attachments/assets/5168e602-0f78-4bab-811e-ccbf2ff36f73)

prediction

![Screenshot 2024-12-14 193628](https://github.com/user-attachments/assets/1b32d32d-a460-49c9-9fb0-5ccd3b90802a)

Upload a plant leaf image in .jpg, .jpeg, or .png format.

Predicts the disease from the uploaded image using a pre-trained TensorFlow model.

4. Real-Time Detection Page

Activates the webcam to detect leaves in real-time using a YOLOv5 model.

Highlights detected leaves with bounding boxes and predicts the disease name.

Displays the results directly on the live camera feed.

![Screenshot 2024-12-14 193712](https://github.com/user-attachments/assets/3ab02127-09c4-4253-a173-2c9dd2972543)

![Screenshot 2024-12-14 193936](https://github.com/user-attachments/assets/ab6ec0f3-c202-4798-a916-a4382f89ed28)

![Screenshot 2024-12-14 194143](https://github.com/user-attachments/assets/02a66b6e-290f-4035-b1fe-c556aaad2cd5)

Requirements
Python 3.7+
TensorFlow
OpenCV
PyTorch
Streamlit

How to Run

Clone the repository:

git clone https://github.com/Eranda724/Plant-Disease-Detection-System

cd plant-disease-detection

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run main.py

Open the local URL provided in your terminal.

Future Enhancements
Add more disease classes to the TensorFlow model.
Improve the real-time detection performance.
Integrate suggestions for managing detected diseases.
