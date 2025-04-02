#!/bin/bash

# Print header
echo "======================================="
echo "   Advanced Deepfake Detector Launcher"
echo "======================================="

# Check if virtual environment exists
if [ ! -d "deepfake_env" ]; then
    echo "Virtual environment not found. Running setup first..."
    chmod +x setup.sh
    ./setup.sh
fi

# Activate virtual environment
source deepfake_env/bin/activate

# Check if activation worked
if [ $? -ne 0 ]; then
    echo "Error: Could not activate virtual environment. Please check that it exists."
    exit 1
fi

# Check and install required dependencies
echo "Checking dependencies..."
missing_deps=0

# Detect system architecture for proper TensorFlow installation
ARCHITECTURE=$(uname -m)
echo "Detected system architecture: $ARCHITECTURE"

# Function to check and install a package
check_and_install() {
    pkg_name=$1
    pkg_import=$2
    
    # Check if package is available
    python -c "import $pkg_import" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing $pkg_name..."
        pip install $pkg_name || pip install --break-system-packages $pkg_name
        if [ $? -ne 0 ]; then
            echo "Warning: Could not install $pkg_name. Some features may not work correctly."
            missing_deps=1
        fi
    else
        echo "✓ $pkg_name is installed"
    fi
}

# Install all required packages
check_and_install "streamlit" "streamlit"
check_and_install "opencv-python" "cv2"
check_and_install "pillow" "PIL"
check_and_install "numpy" "numpy"
check_and_install "matplotlib" "matplotlib"
check_and_install "scikit-learn" "sklearn"
check_and_install "pandas" "pandas"
check_and_install "pygame" "pygame"
check_and_install "urllib3" "urllib.request"

# Special handling for TensorFlow and Keras based on architecture
python -c "import keras" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Keras not found. Attempting installation based on your system architecture..."
    
    if [[ "$ARCHITECTURE" == "arm64" ]]; then
        # For Apple Silicon (M1/M2/M3)
        echo "Detected Apple Silicon. Installing TensorFlow via apple TensorFlow fork..."
        pip install tensorflow-macos || pip install --break-system-packages tensorflow-macos
        
        if [ $? -eq 0 ]; then
            pip install tensorflow-metal || pip install --break-system-packages tensorflow-metal
            pip install keras || pip install --break-system-packages keras
            echo "✓ Installed TensorFlow for Apple Silicon"
        else
            echo "Warning: Could not install TensorFlow for Apple Silicon."
            missing_deps=1
        fi
    else
        # For Intel machines
        echo "Detected Intel architecture. Installing standard TensorFlow..."
        pip install tensorflow keras || pip install --break-system-packages tensorflow keras
        
        if [ $? -ne 0 ]; then
            echo "Trying with tensorflow-cpu as fallback..."
            pip install tensorflow-cpu keras || pip install --break-system-packages tensorflow-cpu keras
            
            if [ $? -ne 0 ]; then
                echo "Warning: Could not install TensorFlow/Keras. The app will use fallback detection methods."
                missing_deps=1
            fi
        fi
    fi
else
    echo "✓ Keras is already installed"
fi

# Check for model files
if [ ! -f "code/PretrainedModel/dffnetv2B0.h5" ]; then
    echo "Model files not unpacked. Unpacking now..."
    cd code/PretrainedModel/
    if [ -f "dffnetv2B0.zip" ]; then
        unzip -o dffnetv2B0.zip
    else
        echo "Warning: Model ZIP file not found. The app will use fallback detection methods."
    fi
    cd ../../
fi

# Check for the pickle file
if [ ! -f "code/PretrainedModel/streamlit_deepfake_detector/deepfake_predictor.pkl" ]; then
    echo "Warning: Prediction model file not found. The app will run in fallback mode."
fi

# Create sounds directory
sounds_dir="code/PretrainedModel/streamlit_deepfake_detector/sounds"
if [ ! -d "$sounds_dir" ]; then
    echo "Creating sounds directory..."
    mkdir -p "$sounds_dir"
fi

# Check for sample images
if [ ! -d "code/PretrainedModel/streamlit_deepfake_detector/images" ]; then
    echo "Creating sample images directory structure..."
    mkdir -p code/PretrainedModel/streamlit_deepfake_detector/images/Real
    mkdir -p code/PretrainedModel/streamlit_deepfake_detector/images/Fake
    echo "Note: You'll need to add sample images for the game mode to work properly."
fi

# Run the application
echo "Launching the Advanced Deepfake Detector..."
cd code/PretrainedModel/streamlit_deepfake_detector
streamlit run final_app.py 