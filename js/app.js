// Initialize Lucide Icons
lucide.createIcons();

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('result');
const webcamBtn = document.getElementById('webcam-btn');
const videoContainer = document.getElementById('video-container');
const video = document.getElementById('webcam');
const captureBtn = document.getElementById('capture-btn');

let currentFile = null;
let stream = null;

// API Base URL
const API_URL = 'http://localhost:8000';

// Handle Drag & Drop
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        videoContainer.style.display = 'none';
        stopWebcam();
    };
    reader.readAsDataURL(file);
}

// Webcam Logic
webcamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        videoContainer.style.display = 'block';
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'none';
    } catch (err) {
        console.error("Webcam Error: ", err);
        alert("Unable to access webcam. Please check permissions.");
    }
});

captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        handleFile(file);
    }, 'image/jpeg');
});

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Analyze API Call
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show loading
    loader.style.display = 'block';
    analyzeBtn.disabled = true;
    resultCard.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();
        displayResult(data);
        fetchHistory();
    } catch (err) {
        console.error(err);
        alert("Error analyzing image. Make sure the backend is running at ${API_URL}");
    } finally {
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function displayResult(data) {
    resultCard.style.display = 'block';
    resultCard.className = 'result-card ' + (data.status === 'Genuine' ? 'result-genuine' : 'result-fake');

    document.getElementById('result-status').innerText = data.status;
    document.getElementById('confidence-badge').innerText = `${data.confidence}% Confidence`;
    document.getElementById('confidence-badge').style.background = data.status === 'Genuine' ? '#10b981' : '#ef4444';
    document.getElementById('confidence-badge').style.color = 'white';

    document.getElementById('result-text').innerText = `Medicine: ${data.details.medicine_name}`;
    document.getElementById('result-details').innerText = data.details.analysis;

    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth' });
}

function resetScanner() {
    currentFile = null;
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    resultCard.style.display = 'none';
    fileInput.value = '';
    fetchHistory();
}

async function fetchHistory() {
    try {
        const response = await fetch(`${API_URL}/history`);
        const data = await response.json();
        const historyList = document.getElementById('history-list');

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.map(file => `
                <div class="history-item" style="flex: 0 0 80px; text-align: center;">
                    <div style="width: 80px; height: 80px; overflow: hidden; border-radius: 8px; margin-bottom: 0.5rem; background: #eee;">
                        <img src="${API_URL}/static/${file}" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                </div>
            `).join('');
        }
    } catch (err) {
        console.warn("History fetch failed:", err);
    }
}

// Initial history fetch
fetchHistory();
