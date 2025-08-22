import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import time

# Set page configuration
st.set_page_config(
    page_title="QuantAI Image Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .uploaded-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Image processing functions
def analyze_image(image):
    """Simulate image analysis and return results"""
    # Convert to numpy array for "analysis"
    img_array = np.array(image)
    
    # Generate some mock analysis results
    results = {
        'width': image.width,
        'height': image.height,
        'aspect_ratio': round(image.width / image.height, 2),
        'format': image.format if hasattr(image, 'format') else 'Unknown',
        'mode': image.mode,
        'estimated_objects': np.random.randint(5, 50),
        'color_variance': round(np.random.uniform(0.1, 0.9), 3),
        'sharpness': round(np.random.uniform(0.3, 0.95), 3),
        'brightness': round(np.random.uniform(0.4, 0.98), 3),
    }
    
    return results

def enhance_image(image, enhancement_type, factor):
    """Apply image enhancements"""
    if enhancement_type == "Brightness":
        enhancer = ImageEnhance.Brightness(image)
    elif enhancement_type == "Contrast":
        enhancer = ImageEnhance.Contrast(image)
    elif enhancement_type == "Sharpness":
        enhancer = ImageEnhance.Sharpness(image)
    elif enhancement_type == "Color":
        enhancer = ImageEnhance.Color(image)
    else:
        return image
    
    return enhancer.enhance(factor)

# Sidebar with clear navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>📊 QuantAI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation options
    st.header("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Image Analysis", "Enhancement", "Batch Processing", "Settings"],
        index=0
    )
    
    st.markdown("---")
    st.header("Analysis Settings")
    
    # Analysis settings
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    detail_level = st.select_slider("Detail Level", options=["Low", "Medium", "High"], value="Medium")
    
    if app_mode == "Enhancement":
        st.markdown("---")
        st.header("Enhancement Settings")
        enhancement_type = st.selectbox(
            "Enhancement Type",
            ["Brightness", "Contrast", "Sharpness", "Color"]
        )
        enhancement_factor = st.slider("Enhancement Factor", 0.5, 2.0, 1.0, 0.1)
    
    st.markdown("---")
    st.header("About")
    st.info("""
    This app uses advanced AI models to analyze and process your images for quantitative insights.
    Upload an image to get started.
    """)

# Main content area
st.markdown('<p class="main-header">QuantAI Image Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Advanced image processing with AI-powered insights</p>', unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📸 Image Upload & Analysis", "🖼️ Enhanced Image", "📈 Data & Export"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Your Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or select files",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="main_uploader"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True, output_format="auto")
            
            # Process button
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing image with AI..."):
                    # Simulate processing time
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    
                    # Analyze the image
                    analysis_results = analyze_image(image)
                    st.session_state.analysis_results = analysis_results
                    st.session_state.original_image = image
                    st.success("Analysis complete!")
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    
                    # Create metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Image Width", f"{analysis_results['width']}px")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Image Height", f"{analysis_results['height']}px")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Aspect Ratio", analysis_results['aspect_ratio'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Color Mode", analysis_results['mode'])
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Recent Analyses")
        
        # Display recent analyses if available
        if 'analysis_results' in st.session_state:
            st.info(f"Last analysis: {st.session_state.analysis_results['width']}x{st.session_state.analysis_results['height']}")
        else:
            st.info("No recent analyses yet")
        
        st.markdown("### Quick Actions")
        
        if st.button("Example Analysis 1", use_container_width=True):
            st.info("This would load a sample analysis")
        
        if st.button("Example Analysis 2", use_container_width=True):
            st.info("This would load another sample analysis")

with tab2:
    if uploaded_file is not None and 'original_image' in st.session_state:
        st.markdown("### Image Enhancement")
        
        enhancement_col1, enhancement_col2 = st.columns(2)
        
        with enhancement_col1:
            enhancement_type = st.selectbox(
                "Enhancement Type",
                ["Brightness", "Contrast", "Sharpness", "Color", "Blur", "Detail"],
                key="enhance_select"
            )
            
            enhancement_factor = st.slider(
                "Enhancement Factor", 
                0.5, 2.0, 1.0, 0.1,
                key="enhance_slider"
            )
            
            if st.button("Apply Enhancement", type="primary"):
                with st.spinner("Applying enhancement..."):
                    if enhancement_type == "Blur":
                        enhanced_image = st.session_state.original_image.filter(
                            ImageFilter.GaussianBlur(radius=enhancement_factor)
                        )
                    else:
                        enhanced_image = enhance_image(
                            st.session_state.original_image, 
                            enhancement_type, 
                            enhancement_factor
                        )
                    
                    st.session_state.enhanced_image = enhanced_image
                    st.success("Enhancement applied!")
        
        with enhancement_col2:
            if 'enhanced_image' in st.session_state:
                st.image(
                    st.session_state.enhanced_image, 
                    caption=f"Enhanced: {enhancement_type} {enhancement_factor}x",
                    use_column_width=True
                )
                
                # Download enhanced image
                buf = io.BytesIO()
                st.session_state.enhanced_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Enhanced Image",
                    data=byte_im,
                    file_name="enhanced_image.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("Apply an enhancement to see the result here")
    else:
        st.warning("Upload and analyze an image first to access enhancement features")

with tab3:
    st.markdown("### Analysis Data & Export")
    
    if 'analysis_results' in st.session_state:
        # Display analysis results as a dataframe
        results_df = pd.DataFrame.from_dict(
            st.session_state.analysis_results, 
            orient='index', 
            columns=['Value']
        )
        st.dataframe(results_df, use_container_width=True)
        
        # Export options
        st.markdown("### Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = results_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="image_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = results_df.to_json()
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="image_analysis.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Generate a simple visualization
            if st.button("Generate Chart", use_container_width=True):
                chart_data = pd.DataFrame({
                    'Attribute': ['Width', 'Height', 'Objects', 'Sharpness'],
                    'Value': [
                        st.session_state.analysis_results['width'],
                        st.session_state.analysis_results['height'],
                        st.session_state.analysis_results['estimated_objects'],
                        st.session_state.analysis_results['sharpness'] * 100
                    ]
                })
                
                st.bar_chart(chart_data.set_index('Attribute'))
    else:
        st.info("Analyze an image to see data and export options")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>QuantAI Image Analysis Platform • v2.1 • "
    "<a href='mailto:support@quantai.com'>Contact Support</a></div>", 
    unsafe_allow_html=True
)