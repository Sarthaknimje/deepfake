# Deepfake Detector

An advanced tool for detecting deepfake images using an ensemble of specialized AI models.

## Features

- **Multi-model Ensemble**: Combines 12 specialized models for more accurate detection
- **Detailed Analysis**: Provides comprehensive breakdown of detection factors
- **Visual Feedback**: Generates heatmaps highlighting potential manipulation areas
- **Game Mode**: Test your ability to spot deepfakes with an interactive game
- **Model Testing**: Compare performance of different detection models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. Create a virtual environment:
   ```
   python -m venv deepfake_env
   source deepfake_env/bin/activate  # On Windows: deepfake_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the App

```
cd deepfake
streamlit run standalone_app.py
```

### Detector Mode

Upload any image to analyze it for potential manipulation:
- Get detailed analysis from multiple specialized models
- View heatmaps highlighting suspicious areas
- See comprehensive breakdown of decision factors

### Game Mode

Test your ability to spot deepfakes:
- Challenge yourself with real and fake images
- Get immediate feedback on your guesses
- Learn about telltale signs of manipulation
- Track your score and accuracy over time

### Model Testing

Evaluate and compare model performance:
- Test against a dataset of known images
- Compare accuracy across different models
- Identify strengths and weaknesses of each approach

## Dataset

The application expects images in the following structure:
```
Dataset/
  ├── Real/        # Real images
  ├── Fake/        # Fake/manipulated images
  └── Test/        # Test images (optional)
      ├── Real/    # Real test images
      └── Fake/    # Fake test images
```

## How It Works

Our deepfake detector analyzes multiple aspects of images:

1. **Noise Pattern Analysis**: Examines noise distributions which differ between real and manipulated images
2. **Edge Consistency**: Checks for unnatural edges that may indicate manipulation
3. **Texture Analysis**: Looks for inconsistencies in texture patterns
4. **Frequency Domain**: Analyzes frequency patterns that are difficult for AI to replicate perfectly

The ensemble combines these insights for a more robust detection than any single model could achieve.

## License

[MIT License](LICENSE)

## Acknowledgments

- All the amazing open-source libraries and frameworks that made this project possible
- The computer vision and deep learning communities for their ongoing research in deepfake detection 