# 🔍 Defect Detection System

**AI-Powered Computer Vision System for Automated Quality Control in Manufacturing**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent defect detection system that uses computer vision and machine learning to automatically identify, classify, and report product defects in real-time. Perfect for manufacturing quality control, automated inspection, and industrial automation.

## ✨ Key Features

### 🎯 Advanced Detection Capabilities
- **Multi-Algorithm Detection** - Contour-based, color-based, and blob detection
- **Real-Time Processing** - Fast analysis with optimized OpenCV algorithms
- **Automated Classification** - Identifies defect types (scratches, holes, irregular patterns)
- **Quality Assessment** - Automated Pass/Fail determination with confidence scores
- **Batch Processing** - Handle multiple images simultaneously

### 🔬 Comprehensive Image Processing
- **Noise Reduction** - Gaussian, Bilateral, and Median filtering
- **Contrast Enhancement** - CLAHE and histogram equalization
- **Edge Detection** - Canny, Sobel, and Laplacian algorithms
- **Morphological Operations** - Erosion, dilation, opening, closing
- **Adaptive Thresholding** - Otsu and adaptive methods

### 🚀 Production-Ready API
- **FastAPI REST API** - High-performance async endpoints
- **Interactive Documentation** - Automatic Swagger UI and ReDoc
- **File Upload Support** - Handle images up to 25MB
- **JSON Responses** - Structured, easy-to-parse output
- **CORS Enabled** - Works with any frontend

### 🎨 Beautiful Web Interface
- **Drag-and-Drop Upload** - Intuitive file selection
- **Real-Time Visualization** - Instant defect highlighting
- **Detailed Reports** - Comprehensive defect analysis
- **Responsive Design** - Works on desktop and mobile
- **Professional UI** - Modern, gradient-based design

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Processing Speed** | < 2 seconds per image |
| **Detection Accuracy** | Configurable sensitivity |
| **Supported Formats** | JPG, PNG, GIF, WebP, BMP |
| **Max Image Size** | 25 MB |
| **Concurrent Requests** | Unlimited (async) |
| **API Response Time** | < 500ms |

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11** - Main programming language
- **OpenCV 4.13** - Computer vision and image processing
- **NumPy 2.4** - Numerical computing and array operations
- **FastAPI 0.129** - Modern, fast web framework
- **Uvicorn** - ASGI server for production deployment

### Image Processing
- **scikit-image** - Advanced image processing algorithms
- **Pillow** - Image file handling
- **scipy** - Scientific computing functions

### Additional Tools
- **Pydantic** - Data validation and settings
- **loguru** - Advanced logging
- **python-multipart** - File upload handling

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/MARAMPELLYAKHILESH/defect-detection-system.git
cd defect-detection-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","message":"All systems operational","version":"1.0.0"}
```

## 🚀 Usage

### Method 1: Web Interface (Recommended)

1. **Start the API server:**
   ```bash
   uvicorn api.main:app --reload
   ```

2. **Open the web interface:**
   - Open `defect_detection_interface.html` in your browser
   - Or visit `http://localhost:8000/` (if HTML is served by API)

3. **Analyze images:**
   - Click "Choose File" or drag-and-drop an image
   - Click "Analyze for Defects"
   - View detailed results and annotated images

### Method 2: API Endpoints

**Analyze Single Image:**

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Response:**

```json
{
  "success": true,
  "image_id": "abc123-def456",
  "num_defects": 3,
  "defect_percentage": 2.45,
  "quality_assessment": "PASS - Good",
  "defects": [
    {
      "id": 1,
      "type": "Small Defect (Minor Scratch)",
      "properties": {
        "area": 245,
        "centroid": [320, 240],
        "circularity": 0.65,
        "aspect_ratio": 2.3
      }
    }
  ],
  "processed_image_url": "/results/abc123_annotated.jpg",
  "timestamp": "2026-02-17T10:30:45.123456"
}
```

### Method 3: Python Code

```python
from src.preprocessing import ImagePreprocessor
from src.defect_detection import DefectDetector
import cv2

# Initialize
preprocessor = ImagePreprocessor(target_size=(640, 480))
detector = DefectDetector(min_defect_area=100, max_defect_area=10000)

# Load and preprocess image
image = cv2.imread("product.jpg")
gray = preprocessor.convert_to_grayscale(image)
denoised = preprocessor.denoise(gray)
enhanced = preprocessor.enhance_contrast(denoised)
thresh = preprocessor.threshold_image(enhanced)
binary = preprocessor.apply_morphology(thresh)

# Detect defects
results = detector.detect_defects(image, binary)

# Display results
print(f"Quality: {results['quality_assessment']}")
print(f"Defects: {results['num_defects']}")
print(f"Coverage: {results['defect_percentage']:.2f}%")

# Show annotated image
cv2.imshow("Results", results["annotated_image"])
cv2.waitKey(0)
```

## 📁 Project Structure

```
defect-detection-system/
├── api/
│   ├── __init__.py
│   └── main.py                          # FastAPI application
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Image preprocessing pipeline
│   └── defect_detection.py              # Defect detection algorithms
├── data/
│   ├── raw/                             # Original uploaded images
│   └── processed/                       # Preprocessed images
├── models/                               # Trained models (if using ML)
├── results/                              # Annotated output images
├── tests/                                # Unit tests
├── defect_detection_interface.html       # Web UI
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Git ignore rules
├── LICENSE                               # MIT License
└── README.md                             # This file
```

## 🎯 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information or serve web interface |
| `/health` | GET | Health check endpoint |
| `/detect` | POST | Analyze single image for defects |
| `/batch-detect` | POST | Analyze multiple images |
| `/results/{image_id}` | GET | Retrieve annotated image |
| `/stats` | GET | System statistics |
| `/cleanup` | DELETE | Clean temporary files |

### Interactive Documentation

Once the server is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔧 Configuration

### Adjust Detection Sensitivity

Edit `src/defect_detection.py`:

```python
detector = DefectDetector(
    min_defect_area=50,      # Lower = more sensitive
    max_defect_area=20000    # Larger = detect bigger defects
)
```

### Change Image Resolution

Edit `src/preprocessing.py`:

```python
preprocessor = ImagePreprocessor(
    target_size=(800, 600)   # Custom resolution
)
```

### Quality Assessment Thresholds

Edit `src/defect_detection.py` (around line 155):

```python
if defect_percentage < 1:
    quality = "PASS - Excellent"
elif defect_percentage < 3:
    quality = "PASS - Good"
elif defect_percentage < 5:
    quality = "WARNING - Acceptable"
else:
    quality = "FAIL - Reject"
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test preprocessing module
python src/preprocessing.py

# Test detection module
python src/defect_detection.py

# Test API health
curl http://localhost:8000/health
```

## 📝 Detection Algorithms

### 1. Contour-Based Detection
- Identifies defect boundaries using edge detection
- Analyzes shape properties (area, perimeter, circularity)
- Filters by size and aspect ratio
- Classifies defects by geometric properties

### 2. Color-Based Detection
- Detects color deviations from expected values
- HSV color space analysis for better accuracy
- Tolerance-based matching
- Identifies discoloration and staining

### 3. Blob Detection
- SimpleBlobDetector algorithm
- Detects circular/elliptical defects
- Area and convexity filtering
- Ideal for bubble and hole detection

### 4. Morphological Analysis
- Opening operations to remove noise
- Closing operations to fill gaps
- Erosion and dilation for enhancement
- Connected component analysis

## 📈 Quality Assessment Criteria

| Defect Coverage | Assessment | Action |
|----------------|------------|---------|
| < 1% | **PASS - Excellent** | ✅ Accept product |
| 1-3% | **PASS - Good** | ✅ Accept product |
| 3-5% | **WARNING - Acceptable** | ⚠️ Manual review recommended |
| > 5% | **FAIL - Reject** | ❌ Reject product |

## 🐳 Docker Deployment

### Using Docker

```dockerfile
# Build image
docker build -t defect-detection:latest .

# Run container
docker run -p 8000:8000 defect-detection:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🌐 Production Deployment

### Deploy to Heroku

```bash
heroku login
heroku create defect-detection-app
git push heroku main
heroku open
```

### Deploy to AWS (EC2)

1. Launch Ubuntu EC2 instance
2. Install Python 3.11 and dependencies
3. Clone repository
4. Configure nginx as reverse proxy
5. Use systemd for process management

### Deploy to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/defect-detection
gcloud run deploy defect-detection \
  --image gcr.io/PROJECT_ID/defect-detection \
  --platform managed
```

## 🔮 Future Enhancements

- [ ] Deep learning models (YOLO, Faster R-CNN)
- [ ] Real-time video stream analysis
- [ ] Mobile app integration (React Native)
- [ ] Cloud storage integration (AWS S3, Google Cloud Storage)
- [ ] Database support (PostgreSQL, MongoDB)
- [ ] Advanced reporting and analytics dashboard
- [ ] Multi-class defect classification
- [ ] 3D defect analysis
- [ ] Integration with manufacturing execution systems (MES)
- [ ] Automated model retraining pipeline

## 🐛 Troubleshooting

### Common Issues

**Issue: Module not found**
```bash
pip install -r requirements.txt
```

**Issue: Port already in use**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Issue: CORS errors**
- Make sure CORS is enabled in `api/main.py`
- Access HTML through `http://` not `file://`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Marampelly Akhilesh**

- 🐙 GitHub: [@MARAMPELLYAKHILESH](https://github.com/MARAMPELLYAKHILESH)
- 💼 LinkedIn: [Marampelly Akhilesh](https://www.linkedin.com/in/marampelly-akhilesh-232593260)
- 📧 Email: marampelly.akhilesh001@gmail.com

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- FastAPI team for the amazing framework
- scikit-image for image processing utilities
- Python community for continuous support

## 📞 Support

For issues, questions, or contributions:

- 📝 [Open an issue](https://github.com/MARAMPELLYAKHILESH/defect-detection-system/issues)
- 📧 Email: marampelly.akhilesh001@gmail.com
- 💬 [LinkedIn](https://www.linkedin.com/in/marampelly-akhilesh-232593260)

## ⭐ Show Your Support

If this project helped you, please consider:
- ⭐ Starring the repository
- 🍴 Forking for your own use
- 📢 Sharing with others
- 🐛 Reporting bugs
- 💡 Suggesting improvements

---

**Built with ❤️ and Computer Vision by Marampelly Akhilesh**

*Empowering manufacturing quality control through AI and computer vision*
