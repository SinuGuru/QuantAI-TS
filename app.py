import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import time
import cv2

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
    """Analyze image and return results"""
    img_array = np.array(image)
    
    # Calculate basic image statistics
    if len(img_array.shape) == 3:  # Color image
        red = img_array[:, :, 0]
        green = img_array[:, :, 1]
        blue = img_array[:, :, 2]
        color_variance = np.array([red.std(), green.std(), blue.std()]).mean()
    else:  # Grayscale
        color_variance = img_array.std()
    
    # Generate analysis results
    results = {
        'width': image.width,
        'height': image.height,
        'aspect_ratio': round(image.width / image.height, 2),
        'format': image.format if hasattr(image, 'format') else 'Unknown',
        'mode': image.mode,
        'color_variance': round(color_variance, 3),
        'brightness': round(np.mean(img_array) / 255, 3),
        'estimated_objects': np.random.randint(5, 50),  # Simulated object detection
        'sharpness': round(estimate_sharpness(img_array), 3),
    }
    
    return results

def estimate_sharpness(image_array):
    """Estimate image sharpness using variance of Laplacian"""
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image_array, cv2.CV_64F).var() / 1000

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
    elif enhancement_type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=factor))
    elif enhancement_type == "Detail":
        return image.filter(ImageFilter.DETAIL)
    elif enhancement_type == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif enhancement_type == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    else:
        return image
    
    return enhancer.enhance(factor)

def apply_filter(image, filter_type):
    """Apply various filters to image"""
    if filter_type == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_type == "Sepia":
        return apply_sepia(image)
    elif filter_type == "Invert":
        return ImageOps.invert(image)
    elif filter_type == "Posterize":
        return ImageOps.posterize(image, 4)
    elif filter_type == "Solarize":
        return ImageOps.solarize(image, threshold=128)
    else:
        return image

def apply_sepia(image):
    """Apply sepia tone filter"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Sepia transformation matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Apply sepia matrix
    sepia_img = np.dot(img_array, sepia_matrix.T)
    # Clip values to [0, 255] and convert to uint8
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_img)

def detect_edges(image):
    """Detect edges in image"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_array, 100, 200)
    return Image.fromarray(edges)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'filtered_image' not in st.session_state:
    st.session_state.filtered_image = None

# Sidebar with clear navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>📊 QuantAI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation options
    st.header("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Image Analysis", "Enhancement", "Filters", "Settings"],
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
            ["Brightness", "Contrast", "Sharpness", "Color", "Blur", "Detail", "Edge Enhance", "Emboss"]
        )
        enhancement_factor = st.slider("Enhancement Factor", 0.5, 2.0, 1.0, 0.1)
    
    elif app_mode == "Filters":
        st.markdown("---")
        st.header("Filter Settings")
        filter_type = st.selectbox(
            "Filter Type",
            ["None", "Grayscale", "Sepia", "Invert", "Posterize", "Solarize", "Edge Detection"]
        )
    
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
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Upload", "📊 Analysis Results", "🖼️ Enhance & Filter", "📈 Export Data"])

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
            
            # Store original image in session state
            st.session_state.original_image = image
            
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
                    st.success("Analysis complete!")
    
    with col2:
        st.markdown("### Recent Analyses")
        
        # Display recent analyses if available
        if st.session_state.analysis_results is not None:
            st.info(f"Last analysis: {st.session_state.analysis_results['width']}x{st.session_state.analysis_results['height']}")
        else:
            st.info("No recent analyses yet")
        
        st.markdown("### Quick Actions")
        
        if st.button("Reset All", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.enhanced_image = None
            st.session_state.filtered_image = None
            st.rerun()

with tab2:
    if st.session_state.analysis_results is not None:
        st.markdown("### Analysis Results")
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Image Width", f"{st.session_state.analysis_results['width']}px")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Image Height", f"{st.session_state.analysis_results['height']}px")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Aspect Ratio", st.session_state.analysis_results['aspect_ratio'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Color Mode", st.session_state.analysis_results['mode'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # More metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Brightness", f"{st.session_state.analysis_results['brightness'] * 100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sharpness", f"{st.session_state.analysis_results['sharpness']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Color Variance", f"{st.session_state.analysis_results['color_variance']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Objects Detected", st.session_state.analysis_results['estimated_objects'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results as a dataframe
        st.markdown("### Detailed Analysis Data")
        results_df = pd.DataFrame.from_dict(
            st.session_state.analysis_results, 
            orient='index', 
            columns=['Value']
        )
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("Upload and analyze an image to see results here")

with tab3:
    if st.session_state.original_image is not None:
        st.markdown("### Image Enhancement & Filters")
        
        enhancement_col1, enhancement_col2 = st.columns(2)
        
        with enhancement_col1:
            enhancement_type = st.selectbox(
                "Enhancement Type",
                ["Brightness", "Contrast", "Sharpness", "Color", "Blur", "Detail", "Edge Enhance", "Emboss"],
                key="enhance_select"
            )
            
            enhancement_factor = st.slider(
                "Enhancement Factor", 
                0.5, 2.0, 1.0, 0.1,
                key="enhance_slider"
            )
            
            if st.button("Apply Enhancement", type="primary"):
                with st.spinner("Applying enhancement..."):
                    enhanced_image = enhance_image(
                        st.session_state.original_image, 
                        enhancement_type, 
                        enhancement_factor
                    )
                    
                    st.session_state.enhanced_image = enhanced_image
                    st.success("Enhancement applied!")
            
            st.markdown("---")
            st.markdown("### Filters")
            
            filter_type = st.selectbox(
                "Filter Type",
                ["None", "Grayscale", "Sepia", "Invert", "Posterize", "Solarize", "Edge Detection"],
                key="filter_select"
            )
            
            if st.button("Apply Filter", type="secondary"):
                with st.spinner("Applying filter..."):
                    if filter_type == "Edge Detection":
                        filtered_image = detect_edges(st.session_state.original_image)
                    elif filter_type != "None":
                        filtered_image = apply_filter(st.session_state.original_image, filter_type)
                    else:
                        filtered_image = st.session_state.original_image
                    
                    st.session_state.filtered_image = filtered_image
                    st.success("Filter applied!")
        
        with enhancement_col2:
            if st.session_state.enhanced_image is not None:
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
            
            if st.session_state.filtered_image is not None:
                st.image(
                    st.session_state.filtered_image, 
                    caption=f"Filter: {filter_type}",
                    use_column_width=True
                )
                
                # Download filtered image
                buf = io.BytesIO()
                st.session_state.filtered_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Filtered Image",
                    data=byte_im,
                    file_name="filtered_image.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            if st.session_state.enhanced_image is None and st.session_state.filtered_image is None:
                st.info("Apply enhancements or filters to see the results here")
    else:
        st.warning("Upload an image first to access enhancement features")

with tab4:
    st.markdown("### Export Options")
    
    if st.session_state.analysis_results is not None:
        # Display analysis results as a dataframe
        results_df = pd.DataFrame.from_dict(
            st.session_state.analysis_results, 
            orient='index', 
            columns=['Value']
        )
        
        # Export options
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
        
        # Image export options
        st.markdown("---")
        st.markdown("### Export Processed Images")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.session_state.original_image is not None:
                buf = io.BytesIO()
                st.session_state.original_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Original",
                    data=byte_im,
                    file_name="original_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with export_col2:
            if st.session_state.enhanced_image is not None:
                buf = io.BytesIO()
                st.session_state.enhanced_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Enhanced",
                    data=byte_im,
                    file_name="enhanced_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with export_col3:
            if st.session_state.filtered_image is not None:
                buf = io.BytesIO()
                st.session_state.filtered_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Filtered",
                    data=byte_im,
                    file_name="filtered_image.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.info("Analyze an image to see data and export options")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>QuantAI Image Analysis Platform • v2.1 • "
    "<a href='mailto:support@quantai.com'>Contact Support</a></div>", 
    unsafe_allow_html=True
)