# 🎨 Paint Animation Web Application

Transform your images into beautiful animated paintings using advanced computer vision and animation techniques. This web application creates stunning animated videos that simulate the painting process with realistic brush strokes and color transitions.

![Paint Animation Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)

## ✨ Features

- **🎨 Image to Animation**: Convert static images into dynamic painting animations
- **🖌️ Realistic Brush Strokes**: Advanced computer vision algorithms create natural painting effects
- **🎯 Smart Color Grouping**: K-means clustering for intelligent color segmentation
- **📱 Responsive Design**: Beautiful, modern UI that works on all devices
- **⚡ Fast Processing**: Optimized algorithms for quick video generation
- **📊 Usage Analytics**: Track application usage with built-in statistics
- **🔒 Secure Upload**: Safe file handling with validation and cleanup
- **📱 PWA Support**: Progressive Web App capabilities for mobile users

## 🚀 Live Demo

[![Railway Deployment](https://railway.app/button.svg)](https://paint-me-production.up.railway.app/)

## 🏗️ Architecture

This application follows a modular architecture with clear separation of concerns:

```
hexa-paint-animation/
├── frontend/          # User interface (HTML, CSS, JavaScript)
├── backend/           # API server (Flask)
├── algorithms/        # Core animation algorithms
├── main.py           # Application entry point
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### 📁 Project Structure

- **Frontend**: Modern, responsive web interface with drag-and-drop upload
- **Backend**: RESTful API with Flask, handling file uploads and video generation
- **Algorithms**: Advanced computer vision and animation processing
- **Database**: SQLite for usage tracking and analytics

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework and API server
- **OpenCV**: Computer vision and image processing
- **scikit-image**: Advanced image analysis
- **NumPy/SciPy**: Numerical computing
- **imageio**: Video generation and processing

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with flexbox and grid
- **JavaScript (ES6+)**: Interactive functionality
- **Progressive Web App**: Offline capabilities

### Infrastructure
- **GitHub Actions**: CI/CD pipeline
- **Docker**: Containerization
- **Heroku/Railway/Render**: Cloud deployment platforms

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmedMedani01/paint-animation.git
   cd paint-animation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t paint-animation .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 paint-animation
   ```

3. **Using Docker Compose**
   ```bash
   docker-compose up
   ```

## 📖 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main application interface |
| `POST` | `/api/upload` | Upload image and generate animation |
| `GET` | `/api/download/<filename>` | Download generated video |
| `GET` | `/api/stats` | Get usage statistics |
| `GET` | `/api/health` | Health check endpoint |

### Upload Request

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/upload
```

### Response Format

```json
{
  "success": true,
  "message": "Video generated successfully",
  "video_filename": "output.mp4",
  "processing_time": 15.2
}
```

## 🎯 Usage

1. **Upload Image**: Drag and drop or click to select an image file
2. **Processing**: Watch the progress bar as your animation is created
3. **Preview**: View your animated video directly in the browser
4. **Download**: Save your creation to your device

### Supported Formats

- **Input**: PNG, JPG, JPEG, GIF, BMP, TIFF (Max 16MB)
- **Output**: MP4 video with H.264 encoding

## 🔧 Configuration

### Environment Variables

```bash
# Optional configuration
MAX_FILE_SIZE=16777216  # 16MB in bytes
UPLOAD_FOLDER=uploads/
OUTPUT_FOLDER=outputs/
DATABASE_URL=sqlite:///app.db
```

### Algorithm Parameters

- **Color Groups**: Number of color clusters (default: 4)
- **Hex Radius**: Hexagon grid size (default: 30)
- **Jitter**: Randomization factor (default: 0.3)
- **Max Size**: Maximum image width (default: 512px)

## 🚀 Deployment

### GitHub Actions

The repository includes automated deployment workflows for:

- **Heroku**: Automatic deployment on push to main branch
- **Railway**: Cloud platform deployment
- **Render**: Static site deployment

### Manual Deployment

#### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

#### Railway
```bash
# Install Railway CLI
railway login
railway deploy
```

#### Render
1. Connect your GitHub repository
2. Select the build command: `pip install -r requirements.txt`
3. Select the start command: `gunicorn main:app`

## 🧪 Testing

```bash
# Run basic tests
python -c "import algorithms.core; print('Core algorithms OK')"
python -c "import backend.app; print('Backend OK')"
python -c "from main import app; print('Main app OK')"

# Test API endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/stats
```

## 📊 Performance

- **Processing Time**: 10-30 seconds for typical images
- **Memory Usage**: ~200MB for 512x512 images
- **Output Quality**: 30 FPS, H.264 encoded MP4
- **Concurrent Users**: Supports multiple simultaneous uploads

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ahmed Medani**
- GitHub: [@AhmedMedani01](https://github.com/AhmedMedani01)
- LinkedIn: [ahmed-medani91](https://www.linkedin.com/in/ahmed-medani91/)
- Upwork: [AhmedMedani](https://www.upwork.com/freelancers/~01b08566d5c43ab2e3)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Flask team for the excellent web framework
- scikit-image contributors for image processing algorithms
- All open-source contributors who made this project possible

## 📈 Roadmap

- [ ] Real-time video streaming
- [ ] Advanced brush stroke customization
- [ ] Batch processing capabilities
- [ ] AI-powered style transfer
- [ ] Mobile app development
- [ ] Social sharing features

---

**Created with ❤️ by Ahmed Medani using advanced computer vision and animation techniques**

[![GitHub stars](https://img.shields.io/github/stars/AhmedMedani01/paint-animation?style=social)](https://github.com/AhmedMedani01/paint-animation)
[![GitHub forks](https://img.shields.io/github/forks/AhmedMedani01/paint-animation?style=social)](https://github.com/AhmedMedani01/paint-animation)