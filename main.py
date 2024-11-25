import streamlit as st
import tensorflow as tf
import numpy as np

def model_pred(test_image):
    model = tf.keras.models.load_model('trained_model.keras')  # Ensure correct path
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "reversed.jpg"  # Ensure correct path
    st.image(image_path, use_container_width=True)
    st.markdown("""
    ## Welcome to the Plant Disease Recognition System!
    This application leverages **Deep Learning** and **Computer Vision** to identify diseases in plants from images of their leaves. 
    Our goal is to help farmers, gardeners, and agricultural experts take **proactive measures** to ensure healthy crops.

    ### 🌟 Key Features:
    - **Fast and Accurate** disease detection.
    - Supports multiple plant species and diseases.
    - User-friendly interface with clear predictions.
    - Offers actionable insights to treat the detected disease.

    ### 🤔 How It Works:
    1. Upload an image of a plant leaf using the **Disease Recognition** section.
    2. The system will analyze the image using a trained machine learning model.
    3. You'll receive a detailed result indicating the disease and possible next steps.

    ### Why Use This App?
    - Protect your crops from spreading diseases.
    - Reduce guesswork and save time.
    - Increase yield and improve plant health efficiently.

    🛠️ Powered by **TensorFlow** and **Streamlit** for an optimal experience.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ## 🌱 What is This App?
    The **Plant Disease Recognition System** is a cutting-edge tool designed to help you identify diseases in plants using image recognition. 
    It’s a fusion of technology and agriculture to promote healthy crop management.

    ## 🎯 Our Mission
    - To empower farmers and agricultural professionals with modern tools.
    - To minimize crop loss and maximize productivity.
    - To support sustainable farming practices worldwide.

    ## 🔬 How Was It Built?
    This app was developed using:
    - **TensorFlow**: For training a powerful neural network model.
    - **Streamlit**: For creating an intuitive and interactive user interface.
    - **Image Processing Techniques**: To analyze and interpret leaf images.

    ## 🤝 Who Can Use It?
    - Farmers looking to identify crop diseases early.
    - Agricultural experts seeking reliable tools for diagnosis.
    - Gardeners wanting healthier plants.

    ## 🌐 Future Enhancements
    - Support for additional plant species and diseases.
    - Integration with real-time weather data for predictive analysis.
    - Providing treatment suggestions based on the detected disease.

    ---
    🛠️ **Developed by:** [Your Name or Team Name]  
    📧 **Contact us at:** your.email@example.com  
    🌍 **Github:** https://github.com/Eranda724?tab=repositories                                                                                                                 
    🌍 **Linkedin:** https://www.linkedin.com/in/eranda-jayasinghe/
    """)

    #Predict Button
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_inage=st.file_uploader("Choose an Images:")
    if(st.button("Show Image")):
        st.image(test_inage,use_column_width=True)
    #Predict. Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            result_index = model_pred(test_image)
            class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
    st.success("It is ()".format(class_name[result_index]))


