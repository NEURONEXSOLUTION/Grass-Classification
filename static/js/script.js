// File upload and preview functionality
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const removeBtn = document.getElementById('removeBtn');
    const uploadForm = document.getElementById('uploadForm');

    // Browse button click - open file dialog
    browseBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    // Upload area click - also open file dialog
    uploadArea.addEventListener('click', function(e) {
        // Only trigger if clicking directly on upload area (not on browse button)
        if (e.target === uploadArea) {
            fileInput.click();
        }
    });

    // File input change - handle file selection
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Remove button click
    removeBtn.addEventListener('click', removeImage);

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        // Only remove class if leaving the upload area
        if (!uploadArea.contains(e.relatedTarget)) {
            uploadArea.classList.remove('drag-over');
        }
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        if (!file) {
            e.preventDefault();
            alert('Please select an image first.');
            return;
        }

        // Add loading state
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        predictBtn.disabled = true;
    });

    function handleFileSelect(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                predictBtn.classList.remove('hidden');
                predictBtn.disabled = false;
                uploadArea.style.display = 'none';
            };
            
            reader.onerror = function() {
                alert('Error reading file. Please try again.');
            };
            
            reader.readAsDataURL(file);
        } else {
            alert('Please select a valid image file (JPEG, PNG, GIF, etc.).');
        }
    }
});

function removeImage() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const predictBtn = document.getElementById('predictBtn');

    // Reset everything
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    predictBtn.classList.add('hidden');
    predictBtn.disabled = true;
    uploadArea.style.display = 'block';
    
    // Reset predict button text
    predictBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Image';
}

function shareResults() {
    if (navigator.share) {
        navigator.share({
            title: 'Grass Classification Result',
            text: `The AI classified this image as: {{ label }} with {{ confidence }}% confidence`,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        const resultText = `Grass Classification Result: {{ label }} ({{ confidence }}% confidence)`;
        navigator.clipboard.writeText(resultText).then(() => {
            alert('Results copied to clipboard!');
        });
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});