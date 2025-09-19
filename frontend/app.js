/**
 * Paint Animation - Frontend JavaScript
 * Handles file upload, progress tracking, and video display
 */

class PaintAnimationApp {
    constructor() {
        this.apiBaseUrl = '/api';
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.resultSection = document.getElementById('resultSection');
        this.resultContent = document.getElementById('resultContent');
        this.statsContainer = document.getElementById('statsContainer');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadStats();
    }
    
    setupEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });
        
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // Drag and drop functionality
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
        
        // Keyboard accessibility
        this.uploadArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.fileInput.click();
            }
        });
    }
    
    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }
        
        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB.');
            return;
        }
        
        this.uploadFile(file);
    }
    
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        this.showProgress();
        this.hideResult();
        
        // Simulate progress for better UX
        this.simulateProgress();
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(data);
            } else {
                // Enhanced error display with detailed information
                this.showDetailedError(data, response.status);
            }
        } catch (error) {
            this.showDetailedError({
                error: 'Network error',
                error_type: 'NETWORK_ERROR',
                status_code: 0,
                details: error.message
            }, 0);
        } finally {
            this.hideProgress();
        }
    }
    
    simulateProgress() {
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            this.progressFill.style.width = progress + '%';
            
            // Update progress text
            if (progress < 30) {
                this.progressText.textContent = 'Analyzing image...';
            } else if (progress < 60) {
                this.progressText.textContent = 'Creating animation frames...';
            } else if (progress < 90) {
                this.progressText.textContent = 'Generating video...';
            } else {
                this.progressText.textContent = 'Finalizing...';
            }
        }, 500);
        
        // Store interval ID for cleanup
        this.progressInterval = progressInterval;
    }
    
    showProgress() {
        this.progressContainer.style.display = 'block';
        this.uploadArea.style.pointerEvents = 'none';
        this.uploadArea.style.opacity = '0.6';
    }
    
    hideProgress() {
        this.progressContainer.style.display = 'none';
        this.uploadArea.style.pointerEvents = 'auto';
        this.uploadArea.style.opacity = '1';
        
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        this.progressFill.style.width = '0%';
    }
    
    showSuccess(data) {
        this.resultSection.style.display = 'block';
        
        this.resultContent.innerHTML = `
            <div class="success">
                ✅ Animation created successfully! Processing time: ${data.processing_time}s
            </div>
            <div class="video-container">
                <video controls autoplay muted loop>
                    <source src="${this.apiBaseUrl}/video/${data.video_filename}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div class="button-container">
                <a href="${this.apiBaseUrl}/download/${data.video_filename}" class="btn">📥 Download Video</a>
                <button onclick="app.createNew()" class="btn">🔄 Create Another</button>
            </div>
        `;
        
        // Update stats
        this.loadStats();
        
        // Scroll to result
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    showError(message) {
        this.resultSection.style.display = 'block';
        
        this.resultContent.innerHTML = `
            <div class="error">
                ❌ ${message}
            </div>
            <button onclick="app.createNew()" class="btn">🔄 Try Again</button>
        `;
        
        // Scroll to result
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    showDetailedError(errorData, statusCode) {
        this.resultSection.style.display = 'block';
        
        const errorType = errorData.error_type || 'UNKNOWN_ERROR';
        const errorMessage = errorData.error || 'An error occurred';
        const details = errorData.details || 'No additional details available';
        const httpStatus = statusCode || errorData.status_code || 'Unknown';
        
        // Get user-friendly error message based on error type
        let userMessage = this.getUserFriendlyMessage(errorType, errorMessage);
        
        this.resultContent.innerHTML = `
            <div class="error-detailed">
                <div class="error-header">
                    <h3>❌ ${userMessage}</h3>
                    <div class="error-meta">
                        <span class="error-type">Type: ${errorType}</span>
                        <span class="error-status">Status: ${httpStatus}</span>
                    </div>
                </div>
                
                <div class="error-details">
                    <h4>📋 Details:</h4>
                    <p>${details}</p>
                </div>
                
                <div class="error-suggestions">
                    <h4>💡 Suggestions:</h4>
                    <ul>
                        ${this.getErrorSuggestions(errorType)}
                    </ul>
                </div>
                
                <div class="error-actions">
                    <button onclick="app.createNew()" class="btn">🔄 Try Again</button>
                    <button onclick="app.showErrorLog('${errorType}', '${errorMessage}', '${details}', '${httpStatus}')" class="btn btn-secondary">🔍 Show Technical Details</button>
                </div>
            </div>
        `;
        
        // Scroll to result
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    getUserFriendlyMessage(errorType, errorMessage) {
        const friendlyMessages = {
            'PROCESSING_FAILED': 'Video generation failed',
            'FILE_NOT_FOUND': 'Required file not found',
            'PERMISSION_ERROR': 'Permission denied',
            'MEMORY_ERROR': 'Insufficient memory',
            'IMPORT_ERROR': 'Missing dependency',
            'NETWORK_ERROR': 'Network connection failed',
            'UNEXPECTED_ERROR': 'Unexpected error occurred'
        };
        
        return friendlyMessages[errorType] || errorMessage;
    }
    
    getErrorSuggestions(errorType) {
        const suggestions = {
            'PROCESSING_FAILED': [
                'Try using a different image format (JPG, PNG)',
                'Ensure the image is not corrupted',
                'Try with a smaller image size',
                'Check if the image has sufficient contrast'
            ],
            'FILE_NOT_FOUND': [
                'The required sprite files may be missing',
                'Try refreshing the page and uploading again',
                'Contact support if the issue persists'
            ],
            'PERMISSION_ERROR': [
                'The server may not have write permissions',
                'Try uploading a different image',
                'Contact support for assistance'
            ],
            'MEMORY_ERROR': [
                'Try using a smaller image (under 2MB)',
                'Reduce image dimensions (under 1024x1024)',
                'Use a simpler image with less detail'
            ],
            'IMPORT_ERROR': [
                'A required library is missing on the server',
                'This is a server-side issue',
                'Contact support for assistance'
            ],
            'NETWORK_ERROR': [
                'Check your internet connection',
                'Try refreshing the page',
                'Wait a moment and try again'
            ],
            'UNEXPECTED_ERROR': [
                'An unexpected error occurred',
                'Try uploading a different image',
                'Contact support with the error details'
            ]
        };
        
        const errorSuggestions = suggestions[errorType] || [
            'Try uploading a different image',
            'Refresh the page and try again',
            'Contact support if the issue persists'
        ];
        
        return errorSuggestions.map(suggestion => `<li>${suggestion}</li>`).join('');
    }
    
    showErrorLog(errorType, errorMessage, details, statusCode) {
        const logWindow = window.open('', '_blank', 'width=600,height=400');
        logWindow.document.write(`
            <html>
                <head>
                    <title>Error Log - Paint Animation</title>
                    <style>
                        body { font-family: monospace; padding: 20px; background: #f5f5f5; }
                        .log-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                        .log-header { background: #ff4444; color: white; padding: 10px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; }
                        .log-item { margin: 10px 0; padding: 10px; background: #f8f8f8; border-left: 4px solid #ff4444; }
                        .log-label { font-weight: bold; color: #333; }
                        .log-value { color: #666; margin-left: 10px; }
                        pre { background: #f0f0f0; padding: 10px; border-radius: 4px; overflow-x: auto; }
                    </style>
                </head>
                <body>
                    <div class="log-container">
                        <div class="log-header">
                            <h2>🔍 Technical Error Details</h2>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">Error Type:</span>
                            <span class="log-value">${errorType}</span>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">Error Message:</span>
                            <span class="log-value">${errorMessage}</span>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">HTTP Status:</span>
                            <span class="log-value">${statusCode}</span>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">Timestamp:</span>
                            <span class="log-value">${new Date().toISOString()}</span>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">User Agent:</span>
                            <span class="log-value">${navigator.userAgent}</span>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">Details:</span>
                            <pre>${details}</pre>
                        </div>
                        
                        <div class="log-item">
                            <span class="log-label">URL:</span>
                            <span class="log-value">${window.location.href}</span>
                        </div>
                    </div>
                </body>
            </html>
        `);
    }
    
    hideResult() {
        this.resultSection.style.display = 'none';
    }
    
    createNew() {
        this.hideResult();
        this.hideProgress();
        this.fileInput.value = '';
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/stats`);
            const data = await response.json();
            
            this.statsContainer.innerHTML = `
                <h3>📊 Usage Statistics</h3>
                <p>Total animations created: <strong>${data.total_uses}</strong> | 
                   Successful: <strong>${data.successful_uses}</strong> | 
                   Success rate: <strong>${data.success_rate}%</strong></p>
            `;
        } catch (error) {
            console.error('Error loading stats:', error);
            this.statsContainer.innerHTML = `
                <h3>📊 Usage Statistics</h3>
                <p>Unable to load statistics</p>
            `;
        }
    }
    
    // Utility methods
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds.toFixed(1)}s`;
        } else {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PaintAnimationApp();
});

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Error handling for unhandled promises
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.app) {
        window.app.showError('An unexpected error occurred. Please try again.');
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.app) {
        window.app.showError('An unexpected error occurred. Please refresh the page.');
    }
});
