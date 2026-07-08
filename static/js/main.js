// Client Side Interaction Scripts - main.js

document.addEventListener('DOMContentLoaded', () => {
    // 1. Theme Toggle System
    const themeToggleBtn = document.getElementById('theme-toggle');
    const rootElement = document.documentElement;
    
    // Check saved theme or default to dark
    const savedTheme = localStorage.getItem('apex-theme') || 'dark';
    rootElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
    
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = rootElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            rootElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('apex-theme', newTheme);
            updateThemeIcon(newTheme);
            window.dispatchEvent(new Event('theme-changed'));
        });
    }
    
    function updateThemeIcon(theme) {
        const icon = document.querySelector('#theme-toggle i');
        if (!icon) return;
        if (theme === 'light') {
            icon.className = 'fas fa-moon';
        } else {
            icon.className = 'fas fa-sun';
        }
    }
    
    // 2. Interactive Input Sliders
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const outputId = slider.getAttribute('data-value-target');
        if (outputId) {
            const output = document.getElementById(outputId);
            if (output) {
                // Initial load value
                updateSliderDisplay(slider, output);
                
                // On sliding change
                slider.addEventListener('input', () => {
                    updateSliderDisplay(slider, output);
                });
            }
        }
    });
    
    function updateSliderDisplay(slider, output) {
        const val = parseFloat(slider.value);
        const formatType = slider.getAttribute('data-format') || 'number';
        
        if (formatType === 'currency') {
            output.textContent = '$' + val.toLocaleString('en-US', {maximumFractionDigits: 0});
        } else if (formatType === 'percentage') {
            output.textContent = (val * 100).toFixed(1) + '%';
        } else {
            output.textContent = val.toString();
        }
    }
    
    // 3. Circular Prediction Probability Gauges
    const gauge = document.querySelector('.gauge-circle');
    if (gauge) {
        const prob = parseFloat(gauge.getAttribute('data-probability') || 0.0);
        const deg = prob * 360;
        const color = gauge.getAttribute('data-color') || '#10b981';
        gauge.style.background = `conic-gradient(${color} ${deg}deg, rgba(255,255,255,0.08) ${deg}deg)`;
    }
    
    // 4. Batch CSV Drag & Drop Uploader
    const dropzone = document.getElementById('drag-drop-zone');
    const fileInput = document.getElementById('file-upload-input');
    const uploaderText = document.getElementById('upload-text-label');
    const formSubmitBtn = document.getElementById('batch-upload-form');
    
    if (dropzone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            }, false);
        });
        
        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileInput.files = files;
                updateFileLabel(files[0].name);
            }
        });
        
        dropzone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                updateFileLabel(fileInput.files[0].name);
            }
        });
    }
    
    function updateFileLabel(name) {
        if (uploaderText) {
            uploaderText.innerHTML = `Selected File: <strong class="text-success">${name}</strong><br>Click 'Process Batch' to run prediction pipeline.`;
        }
    }
    
    // Loading overlay trigger on large form submissions
    const predictionForm = document.getElementById('credit-prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', () => {
            const loaderOverlay = document.getElementById('loader-overlay');
            if (loaderOverlay) {
                loaderOverlay.style.display = 'flex';
            }
        });
    }
});

// Toast system helper
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0 show`;
    toast.role = 'alert';
    toast.ariaLive = 'assertive';
    toast.ariaAtomic = 'true';
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 4000);
}
