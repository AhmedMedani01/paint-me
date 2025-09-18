# 🔧 Backend - Paint Animation API Server

The backend of the Paint Animation application provides a robust RESTful API built with Flask, handling file uploads, video generation, and usage analytics.

## 📁 Structure

```
backend/
├── __init__.py        # Package initialization
├── app.py            # Main Flask application and API routes
├── config.py         # Configuration settings
├── database.py       # Database utilities and models
└── README.md         # This documentation
```

## 🎯 Features

### Core Functionality
- **📤 File Upload**: Secure image upload with validation
- **🎬 Video Generation**: Integration with animation algorithms
- **📊 Usage Analytics**: Track application usage and statistics
- **⬇️ File Download**: Secure video file serving
- **🔒 Security**: Input validation and file sanitization
- **📈 Monitoring**: Health checks and error tracking

### API Endpoints
- **POST /api/upload**: Upload image and generate animation
- **GET /api/download/<filename>**: Download generated video
- **GET /api/stats**: Get usage statistics
- **GET /api/health**: Health check endpoint
- **GET /api/video/<filename>**: Stream video for preview

## 🛠️ Technologies

- **Flask**: Lightweight web framework
- **SQLite**: Embedded database for analytics
- **Werkzeug**: WSGI utilities and file handling
- **UUID**: Unique identifier generation
- **Threading**: Asynchronous processing support

## 🏗️ Architecture

### Application Structure
```python
# Flask Blueprint Architecture
from flask import Blueprint

api_bp = Blueprint('api', __name__)

# Route registration
@api_bp.route('/upload', methods=['POST'])
def upload_image():
    # Handle file upload and processing
    pass
```

### Database Schema
```sql
-- Usage tracking table
CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT,
    file_size INTEGER,
    processing_time REAL,
    success BOOLEAN
);
```

## 📡 API Documentation

### Upload Endpoint

**POST** `/api/upload`

Upload an image file and generate an animated video.

#### Request
```http
POST /api/upload
Content-Type: multipart/form-data

file: [image file]
```

#### Response
```json
{
    "success": true,
    "message": "Video generated successfully",
    "video_filename": "67e50501-55c3-4f0f-800f-ff1bb320c5ba_output.mp4",
    "processing_time": 15.2,
    "file_size": 1048576
}
```

#### Error Response
```json
{
    "success": false,
    "error": "Invalid file format. Please upload PNG, JPG, JPEG, GIF, BMP, or TIFF files.",
    "code": "INVALID_FORMAT"
}
```

### Download Endpoint

**GET** `/api/download/<filename>`

Download a generated video file.

#### Request
```http
GET /api/download/67e50501-55c3-4f0f-800f-ff1bb320c5ba_output.mp4
```

#### Response
- **Success**: File download with appropriate headers
- **Error**: 404 Not Found if file doesn't exist

### Stats Endpoint

**GET** `/api/stats`

Get application usage statistics.

#### Response
```json
{
    "total_uploads": 150,
    "successful_uploads": 142,
    "failed_uploads": 8,
    "average_processing_time": 12.5,
    "total_file_size": 15728640,
    "last_upload": "2024-01-15T10:30:00Z"
}
```

### Health Check

**GET** `/api/health`

Check API server health status.

#### Response
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "uptime": 3600
}
```

## 🔧 Configuration

### Environment Variables
```python
# config.py
import os

class Config:
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # Processing settings
    MAX_PROCESSING_TIME = 300  # 5 minutes
    CLEANUP_INTERVAL = 3600    # 1 hour
```

### Flask Configuration
```python
# app.py
app = Flask(__name__)
app.config.from_object(Config)

# Security headers
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

## 🔒 Security Features

### File Upload Security
```python
def validate_file(file):
    """Validate uploaded file for security and format."""
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return False
    
    # Check MIME type
    allowed_mimes = {
        'image/png', 'image/jpeg', 'image/gif', 
        'image/bmp', 'image/tiff'
    }
    if file.content_type not in allowed_mimes:
        return False
    
    # Check file size
    if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
        return False
    
    return True
```

### Input Sanitization
- **File Validation**: MIME type and extension checking
- **Size Limits**: Maximum file size enforcement
- **Path Traversal**: Prevention of directory traversal attacks
- **SQL Injection**: Parameterized queries for database operations

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@api_bp.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_image():
    # Upload handling with rate limiting
    pass
```

## 📊 Database Management

### Database Utilities
```python
# database.py
class DatabaseManager:
    def __init__(self, db_path='app.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                file_size INTEGER,
                processing_time REAL,
                success BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_usage(self, ip_address, user_agent, file_size, processing_time, success):
        """Log usage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_stats 
            (ip_address, user_agent, file_size, processing_time, success)
            VALUES (?, ?, ?, ?, ?)
        ''', (ip_address, user_agent, file_size, processing_time, success))
        
        conn.commit()
        conn.close()
```

### Analytics Queries
```python
def get_usage_stats():
    """Get comprehensive usage statistics."""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Total uploads
    cursor.execute('SELECT COUNT(*) FROM usage_stats')
    total_uploads = cursor.fetchone()[0]
    
    # Successful uploads
    cursor.execute('SELECT COUNT(*) FROM usage_stats WHERE success = 1')
    successful_uploads = cursor.fetchone()[0]
    
    # Average processing time
    cursor.execute('SELECT AVG(processing_time) FROM usage_stats WHERE success = 1')
    avg_time = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_uploads': total_uploads,
        'successful_uploads': successful_uploads,
        'failed_uploads': total_uploads - successful_uploads,
        'average_processing_time': round(avg_time, 2)
    }
```

## 🚀 Performance Optimization

### Async Processing
```python
import threading
import time

def process_image_async(image_path, output_path, callback):
    """Process image in background thread."""
    def worker():
        try:
            start_time = time.time()
            success = algorithms.core.paint_image(image_path, output_path)
            processing_time = time.time() - start_time
            
            callback(success, processing_time)
        except Exception as e:
            callback(False, 0, str(e))
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
```

### File Cleanup
```python
import os
import time
from datetime import datetime, timedelta

def cleanup_old_files():
    """Remove files older than 24 hours."""
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours
    
    for folder in ['uploads', 'outputs']:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        print(f"Cleaned up: {file_path}")
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_file_info(filename):
    """Cache file information to reduce I/O operations."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'created': stat.st_ctime,
            'exists': True
        }
    return {'exists': False}
```

## 🧪 Testing

### Unit Tests
```python
import unittest
from backend.app import create_app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
    
    def test_health_check(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_upload_invalid_file(self):
        response = self.client.post('/api/upload', data={
            'file': (io.BytesIO(b'not an image'), 'test.txt')
        })
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])
```

### Integration Tests
```python
def test_full_workflow():
    """Test complete upload to download workflow."""
    # Upload image
    with open('test_image.jpg', 'rb') as f:
        response = client.post('/api/upload', data={'file': f})
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] == True
    
    # Download video
    video_filename = data['video_filename']
    response = client.get(f'/api/download/{video_filename}')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'video/mp4'
```

## 📈 Monitoring & Logging

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Configure application logging."""
    if not app.debug:
        file_handler = RotatingFileHandler(
            'logs/app.log', maxBytes=10240, backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')
```

### Error Tracking
```python
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    app.logger.error(f'Unhandled exception: {str(e)}', exc_info=True)
    
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }), 500
```

## 🔄 Deployment

### Production Configuration
```python
# Production settings
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Production file paths
    UPLOAD_FOLDER = '/app/uploads'
    OUTPUT_FOLDER = '/app/outputs'
```

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
```

## 🐛 Troubleshooting

### Common Issues

#### File Upload Failures
- Check file size limits
- Verify file format support
- Ensure sufficient disk space
- Check file permissions

#### Database Errors
- Verify database file permissions
- Check disk space for database
- Ensure proper SQLite installation
- Review database schema

#### Performance Issues
- Monitor memory usage
- Check CPU utilization
- Review file cleanup processes
- Optimize database queries

### Debug Mode
```python
# Enable debug mode for development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📚 API Examples

### cURL Examples
```bash
# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/upload

# Get statistics
curl http://localhost:5000/api/stats

# Health check
curl http://localhost:5000/api/health

# Download video
curl -O http://localhost:5000/api/download/video.mp4
```

### Python Client
```python
import requests

# Upload image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/upload',
        files={'file': f}
    )

data = response.json()
if data['success']:
    print(f"Video generated: {data['video_filename']}")
    
    # Download video
    video_url = f"http://localhost:5000/api/download/{data['video_filename']}"
    video_response = requests.get(video_url)
    
    with open('output.mp4', 'wb') as f:
        f.write(video_response.content)
```

---

**Backend maintained by Ahmed Medani**  
*Part of the Paint Animation Web Application*
