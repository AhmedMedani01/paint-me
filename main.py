#!/usr/bin/env python3
"""
Paint Animation - Main Application
Combines frontend and backend into a single application
"""

import os
import sys
from flask import Flask, render_template, send_from_directory, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend components
from backend.app import api_bp
from backend.database import DatabaseManager

# Initialize main Flask app
app = Flask(__name__, 
            template_folder='frontend',
            static_folder='frontend')

# Configure app
app.secret_key = 'hexa_paint_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize database
db_manager = DatabaseManager()

# Create necessary directories
for folder in ['uploads', 'outputs', 'hands']:
    os.makedirs(folder, exist_ok=True)

# Register backend routes
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    """Serve the main frontend page"""
    try:
        total_uses, successful_uses = db_manager.get_usage_stats()
        return render_template('index.html', 
                             total_uses=total_uses, 
                             successful_uses=successful_uses)
    except Exception as e:
        print(f"Error loading stats: {e}")
        return render_template('index.html', total_uses=0, successful_uses=0)

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend directory"""
    return send_from_directory('frontend', filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'components': {
            'frontend': 'active',
            'backend': 'active',
            'database': 'active'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', total_uses=0, successful_uses=0), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎨 Paint Animation Web Application")
    print("=" * 60)
    print("Starting application...")
    print("Frontend: http://localhost:5000")
    print("API: http://localhost:5000/api")
    print("Health: http://localhost:5000/health")
    print("=" * 60)
    
    # Use ProxyFix for proper handling behind reverse proxies
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
