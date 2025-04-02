import streamlit as st
import os
import sys

# Add the code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "code", "PretrainedModel", "streamlit_deepfake_detector"))

# Import the final_app.py code
try:
    exec(open(os.path.join(os.path.dirname(__file__), "code", "PretrainedModel", "streamlit_deepfake_detector", "final_app.py")).read())
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