# 🧠 Algorithms - Paint Animation Core Engine

The algorithms module contains the core computer vision and animation processing functions that transform static images into dynamic painting animations. This is the heart of the Paint Animation application.

## 📁 Structure

```
algorithms/
├── __init__.py        # Package initialization
├── core.py           # Main algorithm implementations
└── README.md         # This documentation
```

## 🎯 Core Features

### Image Processing
- **🖼️ Smart Resizing**: Aspect ratio preservation with FFmpeg compatibility
- **🎨 Color Segmentation**: K-means clustering for intelligent color grouping
- **🔍 Edge Detection**: Advanced edge detection with morphological operations
- **📐 Object Detection**: Convex hull and binary mask generation

### Animation Generation
- **🖌️ Brush Stroke Simulation**: Realistic painting brush movements
- **🎭 Path Generation**: Random and intelligent painting paths
- **⏱️ Temporal Sequencing**: Frame-by-frame animation construction
- **🎬 Video Encoding**: H.264 MP4 output with optimized settings

## 🛠️ Technology Stack

### Core Libraries
- **OpenCV**: Computer vision and image processing
- **scikit-image**: Advanced image analysis and morphology
- **NumPy/SciPy**: Numerical computing and spatial operations
- **scikit-learn**: Machine learning (K-means clustering)
- **imageio**: Video generation and encoding
- **matplotlib**: Visualization and debugging

### Specialized Tools
- **FFmpeg**: Video encoding and format conversion
- **Pandas**: Data manipulation for path generation
- **tqdm**: Progress tracking for long operations

## 🏗️ Algorithm Architecture

### Main Processing Pipeline
```python
def paint_image(img_path, video_path):
    """
    Main animation function that processes an image and creates an animated video.
    
    Pipeline:
    1. Load and resize image
    2. Create binary mask and edge detection
    3. Generate object masks
    4. Animate binary mask (outline drawing)
    5. Animate color filling with brush strokes
    6. Export final video
    """
```

### Core Functions Overview

#### 1. Image Loading & Preprocessing
```python
def load_and_resize_image(filename: str, max_size: int = 512):
    """
    Load an image, resize it to max_size width while maintaining aspect ratio,
    and ensure both spatial dimensions are even for FFmpeg compatibility.
    """
```

#### 2. Color Segmentation
```python
def group_colors(image, ngroups):
    """
    Get the actual boundary of the internal object using color k-means clustering segmentation.
    Performs image processing operations to get the boundary of the main object.
    """
```

#### 3. Edge Detection
```python
def detect_edges_thick(rgb: np.ndarray, *, dilate_iters: int = 1, kernel_size: int = 3):
    """
    Detect thick edges in an RGB image using Canny edge detection
    with morphological operations for enhanced edge quality.
    """
```

#### 4. Binary Mask Creation
```python
def create_binary_mask(image: np.ndarray, *, dilate_iters: int = 1, kernel_size: int = 3, min_size: int = 1500):
    """
    Create binary mask from image using edge detection and morphological operations.
    """
```

#### 5. Animation Generation
```python
def animate_binary_mask(mask_rgb, edge_map, hull_img, video_writer, pen_png, skip=100, reverse=False):
    """
    Animate the drawing of binary mask edges with a pen sprite.
    """
```

```python
def animate_colors_kmeans_random_paths(image, base_canvas, video_writer, *, brush_png, ngroups=3, hex_radius=30, jitter=0.1):
    """
    Animate color filling using K-means clusters with random path generation.
    """
```

## 🎨 Algorithm Details

### Color Segmentation (K-means Clustering)

The algorithm uses K-means clustering in the LAB color space for better perceptual color grouping:

```python
def group_colors(image, ngroups):
    # Convert to LAB color space for better clustering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img2 = image.reshape((image.shape[0] * image.shape[1], -1))
    
    # Apply K-means clustering
    clt = KMeans(n_clusters=ngroups)
    clt.fit(img2)
    
    # Generate cluster masks
    for i in range(ngroups):
        clu_img = np.zeros((img2.shape[0]))
        clu_img[clt.labels_ == i] = 1
        # Process and return cluster masks
```

**Benefits:**
- Perceptually uniform color grouping
- Automatic color region detection
- Robust to lighting variations
- Configurable number of color groups

### Edge Detection Pipeline

Advanced edge detection with multiple enhancement steps:

```python
def detect_edges_thick(rgb, *, dilate_iters=1, kernel_size=3):
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    
    # Noise reduction
    gray = cv2.bilateralFilter(gray, 7, 30, 7)
    
    # Adaptive thresholding
    med = np.median(gray)
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    
    # Morphological closing
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Dilation for thicker edges
    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges = cv2.dilate(edges, k, iterations=dilate_iters)
```

**Features:**
- Adaptive thresholding based on image statistics
- Bilateral filtering for noise reduction
- Morphological operations for edge enhancement
- Configurable edge thickness

### Hexagonal Grid Meshing

The algorithm uses hexagonal grids for natural painting patterns:

```python
def hexgrid_mesh_indices_2d(image_shape, region_mask, hex_radius=30, jitter=0.1, rng_seed=None):
    """
    Generate hexagonal grid mesh indices for natural painting patterns.
    """
    # Generate hexagonal grid points
    # Apply jitter for natural variation
    # Map to image coordinates
    # Return row and column indices
```

**Advantages:**
- Natural, organic painting patterns
- Configurable grid density
- Random jitter for variation
- Efficient spatial indexing

### Path Generation Algorithms

#### Random Path Generation
```python
def random_path_order(tile_ids, centres, rng=None):
    """
    Generate random painting order for tiles to create natural painting flow.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Shuffle tile order
    shuffled_ids = rng.permutation(tile_ids)
    return shuffled_ids
```

#### Intelligent Path Generation
```python
def intelligent_path_order(tile_ids, centres):
    """
    Generate intelligent painting paths based on spatial relationships.
    """
    # Calculate distances between tile centers
    # Use nearest neighbor or traveling salesman approach
    # Return optimized painting order
```

### Sprite Overlay System

Realistic brush and pen sprite overlay:

```python
def overlay_rgba(canvas: np.ndarray, sprite: np.ndarray, y: int, x: int) -> None:
    """
    Alpha-blend sprite so its top-left corner touches (y, x) in canvas.
    Ideal for pen tip at the sprite's corner.
    """
    # Extract sprite dimensions
    sh, sw = sprite.shape[:2]
    ch, cw = canvas.shape[:2]
    
    # Calculate overlay bounds
    y1, y2 = max(0, y), min(ch, y + sh)
    x1, x2 = max(0, x), min(cw, x + sw)
    
    # Alpha blending
    if sprite.shape[2] == 4:  # RGBA
        alpha = sprite[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = (
            alpha * sprite[:y2-y1, :x2-x1, :3] + 
            (1 - alpha) * canvas[y1:y2, x1:x2]
        )
    else:  # RGB
        canvas[y1:y2, x1:x2] = sprite[:y2-y1, :x2-x1]
```

## ⚙️ Configuration Parameters

### Image Processing Parameters
```python
# Image resizing
max_size = 512              # Maximum image width
aspect_ratio_preserved = True  # Maintain original proportions

# Edge detection
dilate_iters = 1            # Edge dilation iterations
kernel_size = 3             # Morphological kernel size
min_size = 1500             # Minimum object size

# Color segmentation
ngroups = 4                 # Number of color clusters
color_space = 'LAB'         # Color space for clustering
```

### Animation Parameters
```python
# Hexagonal grid
hex_radius = 30             # Hexagon radius in pixels
jitter = 0.3                # Random jitter amount (0-1)

# Animation timing
fps = 30                    # Video frame rate
skip = 100                  # Frame skip for outline animation
hold_frames = 32            # Final frame hold duration

# Sprite settings
pen_scale = 1.5             # Pen sprite scale factor
brush_scale = 1.5           # Brush sprite scale factor
```

### Video Encoding Parameters
```python
# Video settings
codec = "libx264"           # Video codec
pixelformat = "yuv420p"     # Pixel format for compatibility
macro_block_size = 1        # Macro block size
quality = "medium"          # Encoding quality
```

## 🚀 Performance Optimization

### Memory Management
```python
# Efficient array operations
def process_image_efficiently(image):
    # Use in-place operations where possible
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Process in chunks for large images
    chunk_size = 1000
    for i in range(0, image.shape[0], chunk_size):
        chunk = image[i:i+chunk_size]
        # Process chunk
        image[i:i+chunk_size] = processed_chunk
    
    return image
```

### Parallel Processing
```python
import multiprocessing as mp
from functools import partial

def process_clusters_parallel(image, ngroups, n_processes=None):
    """Process color clusters in parallel."""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split work across processes
    with mp.Pool(n_processes) as pool:
        results = pool.map(
            partial(process_single_cluster, image),
            range(ngroups)
        )
    
    return results
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_edge_detection(image_hash, dilate_iters, kernel_size):
    """Cache edge detection results for identical images."""
    # Perform edge detection
    return edges
```

## 🧪 Testing & Validation

### Unit Tests
```python
import unittest
import numpy as np
from algorithms.core import load_and_resize_image, group_colors

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_image_resizing(self):
        """Test image resizing with aspect ratio preservation."""
        resized, _ = load_and_resize_image('test.jpg', max_size=256)
        self.assertEqual(resized.shape[1], 256)  # Width should be 256
        self.assertTrue(resized.shape[0] % 2 == 0)  # Height should be even
    
    def test_color_clustering(self):
        """Test K-means color clustering."""
        clusters = group_colors(self.test_image, ngroups=3)
        self.assertEqual(len(clusters), 3)
        for cluster in clusters:
            self.assertEqual(cluster.shape[:2], self.test_image.shape[:2])
```

### Integration Tests
```python
def test_full_pipeline():
    """Test complete animation pipeline."""
    # Create test image
    test_image = create_test_image()
    
    # Run full pipeline
    success = paint_image('test.jpg', 'output.mp4')
    
    # Verify output
    assert success == True
    assert os.path.exists('output.mp4')
    
    # Check video properties
    cap = cv2.VideoCapture('output.mp4')
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FPS) == 30
    cap.release()
```

### Performance Benchmarks
```python
import time
import psutil

def benchmark_algorithm():
    """Benchmark algorithm performance."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Run algorithm
    result = paint_image('benchmark.jpg', 'benchmark_output.mp4')
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
    print(f"Success: {result}")
```

## 🔧 Debugging & Visualization

### Debug Visualization
```python
def debug_visualization(image, edges, masks, clusters):
    """Create debug visualization of algorithm steps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    
    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    
    # Binary mask
    axes[0, 2].imshow(masks, cmap='gray')
    axes[0, 2].set_title('Binary Mask')
    
    # Color clusters
    for i, cluster in enumerate(clusters):
        axes[1, i].imshow(cluster)
        axes[1, i].set_title(f'Cluster {i+1}')
    
    plt.tight_layout()
    plt.savefig('debug_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Logging
```python
import logging

# Configure algorithm logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def paint_image(img_path, video_path):
    logger.info(f"Starting animation for {img_path}")
    
    try:
        # Process image
        logger.info("Loading and resizing image...")
        image, _ = load_and_resize_image(img_path)
        
        logger.info("Detecting edges...")
        edges = detect_edges_thick(image)
        
        logger.info("Generating color clusters...")
        clusters = group_colors(image, ngroups=4)
        
        logger.info("Creating animation...")
        # Animation code
        
        logger.info(f"Animation complete: {video_path}")
        return True
        
    except Exception as e:
        logger.error(f"Animation failed: {str(e)}")
        return False
```

## 📊 Algorithm Metrics

### Quality Metrics
- **Edge Detection Accuracy**: Percentage of true edges detected
- **Color Segmentation Quality**: Silhouette coefficient for clusters
- **Animation Smoothness**: Frame-to-frame consistency
- **Processing Time**: End-to-end processing duration

### Performance Metrics
- **Memory Usage**: Peak memory consumption
- **CPU Utilization**: Processing efficiency
- **Output Quality**: Video bitrate and compression ratio
- **Scalability**: Performance with different image sizes

## 🔄 Future Enhancements

### Planned Improvements
- [ ] **AI-Powered Style Transfer**: Neural style transfer integration
- [ ] **Advanced Brush Dynamics**: Physics-based brush simulation
- [ ] **Multi-Resolution Processing**: Pyramid-based processing
- [ ] **Real-Time Preview**: Live animation preview
- [ ] **Custom Brush Shapes**: User-defined brush patterns
- [ ] **Animation Presets**: Predefined animation styles

### Research Areas
- **Deep Learning Integration**: CNN-based feature extraction
- **Temporal Coherence**: Improved frame-to-frame consistency
- **Artistic Style Analysis**: Automatic style detection
- **Interactive Editing**: Real-time parameter adjustment

---

**Algorithms maintained by Ahmed Medani**  
*Part of the Paint Animation Web Application*


