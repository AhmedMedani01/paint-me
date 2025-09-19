"""
Backend API for Paint Animation Web Application
Flask-based REST API for handling image uploads and video generation
"""

import os
import sqlite3
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Blueprint
from werkzeug.utils import secure_filename
import sys

# Add algorithms to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from algorithms.core import paint_image

# Initialize Flask Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize database
def init_db():
    """Initialize SQLite database for usage tracking"""
    conn = sqlite3.connect('usage_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_ip TEXT,
            filename TEXT,
            processing_time REAL,
            success BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def record_usage(user_ip, filename, processing_time, success):
    """Record usage statistics in database"""
    conn = sqlite3.connect('usage_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO usage_stats (user_ip, filename, processing_time, success)
        VALUES (?, ?, ?, ?)
    ''', (user_ip, filename, processing_time, success))
    conn.commit()
    conn.close()

def get_usage_stats():
    """Get usage statistics from database"""
    conn = sqlite3.connect('usage_stats.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM usage_stats')
    total_uses = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM usage_stats WHERE success = 1')
    successful_uses = cursor.fetchone()[0]
    conn.close()
    return total_uses, successful_uses

# API Routes
@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and video generation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            input_filename = f"{unique_id}_input.{file_extension}"
            output_filename = f"{unique_id}_output.mp4"
            
            # Save uploaded file
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            file.save(input_path)
            
            # Process the image
            start_time = time.time()
            print(f"🚀 Starting video generation for {filename}")
            print(f"📁 Input path: {input_path}")
            print(f"📁 Output path: {output_path}")
            
            success = paint_image(input_path, output_path)
            processing_time = time.time() - start_time
            
            # Record usage
            user_ip = request.remote_addr or 'unknown'
            record_usage(user_ip, filename, processing_time, success)
            
            if success and os.path.exists(output_path):
                print(f"✅ Video generation successful: {output_filename}")
                # Clean up input file
                os.remove(input_path)
                return jsonify({
                    'success': True,
                    'video_filename': output_filename,
                    'processing_time': round(processing_time, 2)
                })
            else:
                print(f"❌ Video generation failed for {filename}")
                print(f"📁 Output file exists: {os.path.exists(output_path)}")
                # Clean up files on failure
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
                return jsonify({
                    'error': 'Video generation failed',
                    'error_type': 'PROCESSING_FAILED',
                    'status_code': 500,
                    'details': 'The image processing algorithm failed to generate a video. This could be due to image format issues, insufficient memory, or missing dependencies.'
                }), 500
                
        except FileNotFoundError as e:
            print(f"❌ File not found error: {str(e)}")
            return jsonify({
                'error': 'File not found',
                'error_type': 'FILE_NOT_FOUND',
                'status_code': 404,
                'details': f'Required file not found: {str(e)}'
            }), 404
            
        except PermissionError as e:
            print(f"❌ Permission error: {str(e)}")
            return jsonify({
                'error': 'Permission denied',
                'error_type': 'PERMISSION_ERROR',
                'status_code': 403,
                'details': f'Permission denied: {str(e)}'
            }), 403
            
        except MemoryError as e:
            print(f"❌ Memory error: {str(e)}")
            return jsonify({
                'error': 'Insufficient memory',
                'error_type': 'MEMORY_ERROR',
                'status_code': 507,
                'details': 'The image is too large or complex for processing. Try using a smaller image.'
            }), 507
            
        except ImportError as e:
            print(f"❌ Import error: {str(e)}")
            return jsonify({
                'error': 'Missing dependency',
                'error_type': 'IMPORT_ERROR',
                'status_code': 500,
                'details': f'Required library not available: {str(e)}'
            }), 500
            
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': 'Unexpected error occurred',
                'error_type': 'UNEXPECTED_ERROR',
                'status_code': 500,
                'details': f'{type(e).__name__}: {str(e)}'
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@api_bp.route('/download/<filename>')
def download_file(filename):
    """Handle video file downloads"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=f'animated_{filename}')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@api_bp.route('/video/<filename>')
def serve_video(filename):
    """Serve video files for embedding"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Video error: {str(e)}'}), 500

@api_bp.route('/stats')
def stats():
    """Get usage statistics"""
    total_uses, successful_uses = get_usage_stats()
    return jsonify({
        'total_uses': total_uses,
        'successful_uses': successful_uses,
        'success_rate': round((successful_uses / total_uses * 100) if total_uses > 0 else 0, 2)
    })

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Error handlers
@api_bp.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@api_bp.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@api_bp.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Create Flask app for standalone backend usage
def create_backend_app():
    """Create standalone backend Flask app"""
    app = Flask(__name__)
    app.secret_key = 'hexa_paint_secret_key_2024'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

if __name__ == '__main__':
    print("🎨 Paint Animation Backend API")
    print("=" * 50)
    print("Starting API server...")
    print("API endpoints:")
    print("  POST /api/upload - Upload image and generate video")
    print("  GET  /api/download/<filename> - Download video")
    print("  GET  /api/video/<filename> - Serve video")
    print("  GET  /api/stats - Get usage statistics")
    print("  GET  /api/health - Health check")
    print("=" * 50)
    
    backend_app = create_backend_app()
    backend_app.run(debug=True, host='0.0.0.0', port=5000)
