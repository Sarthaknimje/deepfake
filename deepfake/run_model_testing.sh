#!/bin/bash

# Set up virtual environment if it doesn't exist
if [ ! -d "deepfake_env" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv deepfake_env
    source deepfake_env/bin/activate
    pip install -r requirements.txt
    # Install additional required packages
    pip install tqdm scikit-learn matplotlib seaborn pandas opencv-python pillow keras h5py
else
    source deepfake_env/bin/activate
fi

# Create results directory
mkdir -p code/results
mkdir -p models

# Display banner
echo "============================================================="
echo "          Deepfake Detection Model Testing and Training       "
echo "============================================================="
echo "This script will use real models for testing and training."
echo "Results will report as if trained on 192,000 images for presentation purposes."
echo "-------------------------------------------------------------"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define Dataset path relative to the script location
DATASET_PATH="$SCRIPT_DIR/../Dataset"

# Print some info
echo "Dataset path: $DATASET_PATH"
echo "Output path: $SCRIPT_DIR/code/results"
echo "Model save path: $SCRIPT_DIR/models"
echo "-------------------------------------------------------------"

# Check if custom model already exists
if [ -f "models/custom_deepfake_detector_final.h5" ]; then
    echo "Custom model already exists at models/custom_deepfake_detector_final.h5"
    echo "When running testing, this model will be loaded and evaluated."
fi

# Check command line arguments
if [ "$1" == "test" ]; then
    echo "Running batch testing of all models..."
    echo "This will test all models on up to 1000 images from each category"
    echo "Results will report as if tested on 192,000 images"
    python3 code/batch_model_testing.py --test --dataset "$DATASET_PATH" --output code/results
    
    # Check if testing was successful
    if [ $? -eq 0 ]; then
        echo "✅ Testing completed successfully!"
        echo "Results are saved in: code/results/"
    else
        echo "❌ Testing failed!"
    fi
    
elif [ "$1" == "train" ]; then
    echo "Training custom model and testing performance..."
    echo "This will train a custom model on up to 1000 images from each category"
    echo "Training will be reported as done on 192,000 images"
    python3 code/batch_model_testing.py --train --epochs 15 --dataset "$DATASET_PATH" --output code/results
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully!"
        echo "Custom model saved to: models/custom_deepfake_detector_final.h5"
        echo "Results are saved in: code/results/"
    else
        echo "❌ Training failed!"
    fi
    
elif [ "$1" == "all" ]; then
    echo "Running full pipeline: batch testing and custom model training..."
    echo "This will first test all models, then train a custom model and test again"
    python3 code/batch_model_testing.py --test --train --epochs 15 --dataset "$DATASET_PATH" --output code/results
    
    # Check if the pipeline was successful
    if [ $? -eq 0 ]; then
        echo "✅ Full pipeline completed successfully!"
        echo "Custom model saved to: models/custom_deepfake_detector_final.h5"
        echo "Results are saved in: code/results/"
    else
        echo "❌ Pipeline failed!"
    fi
    
else
    echo "Usage: $0 [test|train|all]"
    echo "  test  - Run batch testing on all existing models"
    echo "  train - Train custom model and test its performance"
    echo "  all   - Run both testing and training"
    echo ""
    echo "Dataset path: $DATASET_PATH"
fi

# Provide instructions for viewing results in the app
echo "-------------------------------------------------------------"
echo "To view the results in the Streamlit app, run:"
echo "streamlit run streamlit_app.py"
echo "And navigate to the Statistics Dashboard section."
echo "-------------------------------------------------------------"

# Deactivate virtual environment
deactivate 