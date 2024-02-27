import streamlit as st
import cv2
import os
from PIL import Image
import joblib
import numpy as np
from io import BytesIO
import tempfile

model = joblib.load('model.pkl')

pca = joblib.load('pca.joblib')



def preprocess_and_crop_image(face_classifier: cv2.CascadeClassifier, image, target_size=(64, 64), x=100, y=100, crop_width=800, crop_height=800):
    # Load the image using OpenCV
    img = cv2.imread(image)

    # Convert the image to grayscale if it's in color
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, crop and save them
    if len(faces) > 0:
        # Create a copy of the original image
        img_copy = img.copy()
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Crop the detected face
            cropped_face = gray_img[y:y + h, x:x + w]

            # Resize the cropped face to the target size
            cropped_face = cv2.resize(cropped_face, target_size)

            # Normalize the cropped face to have values in the range [0, 1]
            cropped_face = cropped_face.astype('float32') / 255.0

        return cropped_face

    # If no faces are detected
    else:
        print(f"No face detected on {image}, so do a normal crop")

        # Crop the image
        cropped_img = gray_img[y:y + crop_height, x:x + crop_width]

        # Resize the cropped image to the target size
        cropped_img = cv2.resize(cropped_img, target_size)

        # Normalize the image to have values in the range [0, 1]
        cropped_img = cropped_img.astype('float32') / 255.0

        return cropped_img

def predict(image_path):
    # Preprocess the input image
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    preprocessed_image = preprocess_and_crop_image(face_classifier, image_path)

    # Reshape the cropped face to match the shape expected by SVM
    preprocessed_image_flattened = preprocessed_image.flatten()

    # Apply PCA transformation
    preprocessed_image_pca = pca.transform([preprocessed_image_flattened])

    # Make predictions using your model
    prediction = model.predict(preprocessed_image_pca)

    return prediction


def main():
    st.title('Your Streamlit App Title')
    st.sidebar.title('Sidebar Title')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image file content
        # image_content = uploaded_file.read()

        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, 'uploaded_image.jpg')

         # Save the uploaded image to the temporary directory
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

        # Display the uploaded image
        image = Image.open(temp_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Perform prediction
        prediction = predict(temp_path)

        # Close the temporary directory
        temp_dir.cleanup()

        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
