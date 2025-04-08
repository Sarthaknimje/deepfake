# Deepfake Detection Model Testing and Training

This module provides functionality to:
1. Batch test all 11 models against images from a test dataset
2. Train a custom 12th model on dataset images
3. Integrate the custom model into the main application

## Prerequisites

- Python 3.7+
- Required Python packages (installed automatically by the script)
- Dataset organized in the following structure:
  ```
  Dataset/
  ├── Test/
  │   ├── Real/
  │   └── Fake/
  ├── Train/
  │   ├── Real/
  │   └── Fake/
  └── Validation/
      ├── Real/
      └── Fake/
  ```

## Usage

### Quick Start

Use the provided bash script to run tests and train models:

```bash
# Run batch testing on all models
./run_model_testing.sh test

# Train a custom model and test it
./run_model_testing.sh train

# Run both testing and training
./run_model_testing.sh all
```

### Manual Execution

You can also run the Python script directly:

```bash
# Activate the virtual environment
source deepfake_env/bin/activate

# Run batch testing on all models
python code/batch_model_testing.py --test --dataset Dataset --output code/results

# Train a custom model
python code/batch_model_testing.py --train --epochs 15 --dataset Dataset --output code/results

# Run both testing and training
python code/batch_model_testing.py --test --train --epochs 15 --dataset Dataset --output code/results
```

## Results and Output

The results will be saved in the `code/results` directory:

- `model_comparison_results.csv`: Metrics for all models
- `model_metrics_comparison.png`: Visualization of model performance
- `/confusion_matrices/`: Individual confusion matrices for each model
- `custom_model_training_history.png`: Training history for the custom model
- `model_comparison_with_custom_model.csv`: Updated metrics including the custom model

## Model Descriptions

The system tests the following 11 models plus a 12th custom-trained model:

1. **EfficientNet_v2B0**: General deepfake detection, highest weight in ensemble
2. **ResNet50_FT**: Facial manipulation detection
3. **DenseNet121_Custom**: GAN artifact detection
4. **VGG16_EdgeAnalysis**: Edge inconsistency detection
5. **Xception_Noise**: Noise pattern analysis
6. **InceptionV3_Frequency**: Frequency domain analysis
7. **CLIP_Visual**: Semantic consistency
8. **MobileNetV3_Texture**: Texture coherence
9. **Vision_Transformer**: Global structure analysis
10. **DINO_SelfSupervised**: Self-supervised features
11. **LightCNN_Forensics**: Digital forensics markers
12. **Custom_Trained**: Custom model trained on specific dataset

## Performance Metrics

For each model, the following metrics are calculated:
- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: Visualization of true/false positives/negatives

## Custom Model Training

The custom model uses an EfficientNetV2B0 base with:
- Transfer learning from ImageNet weights
- Data augmentation for training samples
- Custom classification head
- Two-phase training (feature extraction followed by fine-tuning)

After training, the model is automatically integrated into the ensemble with appropriate weight.

## Integrating Custom Model in Main App

The custom model is automatically detected and integrated into the main Streamlit application. When you run the app after training a custom model, you'll see "Custom model loaded successfully!" in the sidebar.

## Troubleshooting

- If training fails, check that your dataset follows the required structure
- For memory errors, reduce batch size in the code
- If testing is too slow, reduce the number of test images in `batch_test_models()` 