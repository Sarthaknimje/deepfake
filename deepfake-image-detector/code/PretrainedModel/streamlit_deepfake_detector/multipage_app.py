# -------------------
# IMPORTS
# -------------------
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time

# Set page config first
st.set_page_config(layout="wide")

# Add enhanced custom CSS with team branding
st.markdown("""
<style>
    /* Modern gradient background with animation */
    .main {
        background: linear-gradient(-45deg, #ff4b4b, #7e56ff, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Team branding banner */
    .team-banner {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .team-text {
        color: #fff;
        font-size: 18px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
    }
    
    /* Enhanced title with 3D effect */
    .title-text {
        background: linear-gradient(45deg, #ff4b4b, #7e56ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 60px;
        font-weight: 800;
        text-align: center;
        padding: 25px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        transform: perspective(500px) rotateX(10deg);
    }
    
    /* Animated cards */
    .stMetric {
        background: rgba(255,255,255,0.9);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stMetric:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 45px rgba(0,0,0,0.2);
    }

    /* Rainbow progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff4b4b, #7e56ff, #23a6d5, #23d5ab);
        background-size: 300% 300%;
        animation: gradient 5s ease infinite;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# MAIN
# -------------------
def main():
    st.markdown("<h1 class='title-text'>Deepfake Detector 🔍</h1>", unsafe_allow_html=True)
    
    # Add team banner
    st.markdown("""
        <div class='team-banner'>
            <p class='team-text'>Created with ❤️ by Team Pioneers</p>
            <p style='color: #fff; font-size: 14px;'>Advancing AI Safety Through Innovation</p>
        </div>
    """, unsafe_allow_html=True)

# function to load and cache pretrained model
@st.cache_resource()
def load_model():
    path = "../dffnetv2B0"
    # Import required modules
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import Dense, Input, Rescaling
    
    try:
        # Recreate the exact model architecture
        model = Sequential()
        effnet = EfficientNetV2B0(
            weights='imagenet',
            input_shape=(256, 256, 3),
            include_top=False,
            pooling='max',
            classes=2,
            include_preprocessing=True
        )
        model.add(effnet)
        model.add(Dense(1, activation='sigmoid'))
        
        # Load the weights
        model.load_weights(path + '.h5')
        
        # Compile with same configuration as training
        model.compile(
            optimizer='adam',
            loss='bce',
            metrics=['accuracy']
        )
        return model
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Enhanced loading animation
def load_model_with_animation():
    with st.spinner('🤖 Loading AI Model...'):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        model = load_model()
        st.success('✅ Model loaded successfully!')
        time.sleep(1)
        progress.empty()
        return model

# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]
    
    if predicted_prob >= 0.5:
        return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"

# generate selection of sample images 
@st.cache_data()
def load_images():
  real_images = ["images/Real/" + x for x in os.listdir("images/Real/")]
  fake_images = ["images/Fake/" + x for x in os.listdir("images/Fake/")]
  image_library = real_images + fake_images
  image_selection = np.random.choice(image_library, 20, replace=False)

  return image_selection

# Initialize the app with animation
classifier = load_model_with_animation()
images = load_images()

def check_answer(user_guess, true_label, model_prediction):
    model_guess = "Real" if "Real" in model_prediction else "Fake"
    
    # Update scores
    st.session_state.total_games += 1
    
    if user_guess == true_label:
        st.session_state.user_score += 1
        st.session_state.streak += 1
        st.balloons()
    else:
        st.session_state.streak = 0
        
    if model_guess == true_label:
        st.session_state.model_score += 1
    
    # Show results with animations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Your guess: {user_guess}")
    st.markdown(f"### AI's guess: {model_prediction}")
    st.markdown(f"### Correct answer: {true_label}")
    
    if user_guess == true_label:
        st.success("🎉 You got it right!")
    else:
        st.error("❌ Better luck next time!")

def game_mode():
    # Initialize session state variables
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    if 'model_score' not in st.session_state:
        st.session_state.model_score = 0
    if 'total_games' not in st.session_state:
        st.session_state.total_games = 0
    if 'streak' not in st.session_state:
        st.session_state.streak = 0

    st.markdown("""
        <h2 style='text-align: center; background: linear-gradient(45deg, #ff4b4b, #7e56ff);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 40px; margin-bottom: 30px;'>
            🎮 Challenge the AI!
        </h2>
    """, unsafe_allow_html=True)
    
    # Stats display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(45deg, #ff4b4b, #ff8f8f); padding: 20px; border-radius: 15px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Your Score 🏆</h3>
                <h2 style='color: white; margin: 10px 0;'>{st.session_state.user_score}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(45deg, #7e56ff, #23a6d5); padding: 20px; border-radius: 15px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>AI Score 🤖</h3>
                <h2 style='color: white; margin: 10px 0;'>{st.session_state.model_score}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style='background: linear-gradient(45deg, #23a6d5, #23d5ab); padding: 20px; border-radius: 15px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Games Played 🎲</h3>
                <h2 style='color: white; margin: 10px 0;'>{st.session_state.total_games}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div style='background: linear-gradient(45deg, #ff4b4b, #7e56ff); padding: 20px; border-radius: 15px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>Win Streak 🔥</h3>
                <h2 style='color: white; margin: 10px 0;'>{st.session_state.streak}</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Image selection with enhanced styling
    selected_image = image_select(
        "Select an image to analyze:", 
        images,
        return_value="index",
        use_container_width=True
    )
    
    prediction = get_prediction(classifier, images[selected_image])
    true_label = 'Fake' if 'fake' in images[selected_image].lower() else 'Real'

    st.markdown("<h3 style='text-align: center; color: #7e56ff;'>Make your guess!</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("It's Real", use_container_width=True):
            user_guess = "Real"
            check_answer(user_guess, true_label, prediction)
    with col2:
        if st.button("It's Fake", use_container_width=True):
            user_guess = "Fake"
            check_answer(user_guess, true_label, prediction)

def detector_mode():
    st.markdown("<h2 style='text-align: center; color: #7e56ff;'>🔍 Detector Mode</h2>", 
                unsafe_allow_html=True)
    
    # File uploader with preview
    uploaded_file = st.file_uploader(
        "Upload an image to analyze:", 
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, use_container_width=True, caption="Uploaded Image")
        with col2:
            with st.spinner("🔍 Analyzing image..."):
                prediction = get_prediction(classifier, uploaded_file)
                confidence = float(prediction.split(':')[1])
                
                st.markdown("### Analysis Results")
                if "Real" in prediction:
                    st.success(f"✅ {prediction}")
                else:
                    st.error(f"❌ {prediction}")
                
                # Confidence meter with improved styling
                st.markdown("""
                    <style>
                        .stProgress > div > div {
                            background-color: #7e56ff;
                        }
                    </style>""", 
                    unsafe_allow_html=True
                )
                st.progress(confidence)
                st.markdown(f"*Confidence level: {confidence:.2%}*")

page = st.sidebar.selectbox('Select Mode',['Detector Mode','Game Mode']) 

if page == 'Game Mode':
  game_mode()
else:
  detector_mode()

# -------------------
# SCRIPT/MODULE CHECKER
# -------------------
if __name__ == "__main__":
    main()










