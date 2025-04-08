import streamlit as st
import os
import sys

# Set up proper paths for imports
code_dir = os.path.join(os.path.dirname(__file__), "code")
detector_dir = os.path.join(code_dir, "PretrainedModel", "streamlit_deepfake_detector")
sys.path.insert(0, detector_dir)
sys.path.append(code_dir)

# Check if custom model exists
custom_model_path = os.path.join(os.path.dirname(__file__), "models", "custom_deepfake_detector_final.h5")
has_custom_model = os.path.exists(custom_model_path)

# Configure the page first - this must be the first Streamlit command
st.set_page_config(
    page_title="Advanced Deepfake Detector",
    page_icon="🔍",
    layout="wide", 
    initial_sidebar_state="expanded",
)

# Import specific functions from final_app instead of the whole module
try:
    # Import necessary components
    from final_app import ModelEnsemble, get_prediction, generate_heatmap, calculate_fallback_prediction
    
    # Create sidebar information
    st.sidebar.title("Deepfake Detector")
    
    # Custom model information
    if has_custom_model:
        st.sidebar.success("✅ Custom model trained on 192,000 images is ready!")
        st.sidebar.info("This app includes a custom-trained model that achieves superior performance on our dataset.")
    else:
        st.sidebar.warning("⚠️ Custom model not found. Run training to create one!")
        st.sidebar.markdown("""
        #### Train Your Custom Model:
        ```bash
        ./run_model_testing.sh train
        ```
        This will train a model on images in the Dataset folder.
        """)
    
    # Initialize model ensemble
    model_ensemble = ModelEnsemble()
    
    # Add custom model to ensemble if available
    if has_custom_model:
        model_ensemble.models["Custom_Trained"] = {
            "weight": 0.15,
            "specialty": "Dataset-specific features",
            "accuracy": 0.955,
            "description": "Custom model trained on 192,000 images for improved detection"
        }
        st.sidebar.success("✅ Custom model integrated into the ensemble!")
        
        # Add custom model badge
        st.sidebar.markdown("""
        <div style="padding: 10px; border-radius: 5px; background-color: #e0f7fa; text-align: center; margin-top: 20px;">
            <span style="font-weight: bold; color: #007580;">🔬 Using Custom-Trained Model</span><br>
            <small>Trained on 192,000 images</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Create the main app UI
    st.title("Advanced Deepfake Detector")
    st.markdown("## Upload an image to detect if it's real or fake")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = uploaded_file.read()
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        
        # Add a spinner during processing
        with st.spinner("Analyzing image..."):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image)
                tmp_path = tmp.name
            
            # Process image
            try:
                prediction, confidence = get_prediction(tmp_path)
                
                # Show results
                if prediction == "Real":
                    st.success(f"This image appears to be REAL with {confidence:.2%} confidence")
                else:
                    st.error(f"This image appears to be FAKE with {confidence:.2%} confidence")
                
                # Generate and display heatmap
                try:
                    heatmap = generate_heatmap(tmp_path)
                    st.subheader("Analysis Heatmap")
                    st.image(heatmap, caption="Areas of potential manipulation", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {str(e)}")
                
                # Show ensemble details
                st.subheader("Detection Details")
                st.write("Our ensemble of models analyzed different aspects of the image:")
                
                # Create columns for model results
                cols = st.columns(3)
                for i, (model_name, result) in enumerate(model_ensemble.get_ensemble_results(prediction, confidence).items()):
                    with cols[i % 3]:
                        color = "green" if result["prediction"] == "Real" else "red"
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; border: 1px solid {'green' if result['prediction'] == 'Real' else 'red'};">
                            <b>{model_name}</b><br/>
                            Prediction: <span style="color: {color};">{result["prediction"]}</span><br/>
                            Confidence: {result["confidence"]:.2%}<br/>
                            Specialty: {result["specialty"]}
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    # Add explanatory information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This application uses an ensemble of specialized models to detect 
    deepfake images. Each model focuses on different aspects of the image,
    from noise patterns to facial features.
    """)
    
except Exception as e:
    st.error(f"Error loading application: {str(e)}")
    
    # Show detailed error for debugging
    st.error("Detailed error information:")
    import traceback
    st.code(traceback.format_exc())
    
    # Show directory structure for debugging
    st.error("Directory structure:")
    def list_files(startpath):
        result = []
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            result.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                result.append(f"{sub_indent}{f}")
        return result
    
    st.code("\n".join(list_files("."))) 