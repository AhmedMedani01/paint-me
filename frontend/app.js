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
                this.showError(data.error || 'An error occurred while processing your image.');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
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
