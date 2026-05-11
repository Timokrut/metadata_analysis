let currentFileId = null;
let analysisResults = {
    metadata: null,
    video: null,
    audio: null
};

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
    setupFileUpload();
    checkSystemHealth();
});

// Настройка загрузки файла
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
        uploadArea.style.transform = 'translateY(-5px)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.transform = 'translateY(0)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.transform = 'translateY(0)';
        
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Обработка загрузки файла
async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Показываем индикатор загрузки
        showUploadProgress();
        
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentFileId = result.file_id;
            showFileInfo(file, result);
            showAnalysisControls();
            resetResults();
        } else {
            alert('Ошибка при загрузке файла: ' + result.error);
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Ошибка при загрузке файла. Проверьте подключение.');
    } finally {
        hideUploadProgress();
    }
}

// Показать информацию о файле
function showFileInfo(file, uploadResult) {
    const fileInfo = document.getElementById('fileInfo');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileId').textContent = currentFileId;
    fileInfo.style.display = 'block';
}

// Показать панель управления анализом
function showAnalysisControls() {
    document.getElementById('analysisControls').style.display = 'block';
    document.getElementById('analysisControls').scrollIntoView({ behavior: 'smooth' });
}

// Сбросить результаты
function resetResults() {
    analysisResults = {
        metadata: null,
        video: null,
        audio: null
    };
    
    // Сбросить отображение
    ['metadata', 'video', 'audio'].forEach(service => {
        updateServiceDisplay(service, 'waiting', '-', 'Нажмите кнопку для запуска анализа');
        document.getElementById(`${service}Time`).textContent = '';
    });
    
    // Скрыть итоговый результат
    document.getElementById('finalResult').style.display = 'none';
}

// Запуск анализа метаданных
async function analyzeMetadata() {
    if (!currentFileId) {
        alert('Сначала загрузите файл');
        return;
    }
    
    await runAnalysis('metadata', `/analyze/metadata/${currentFileId}`);
}

// Запуск видеоанализа
async function analyzeVideo() {
    if (!currentFileId) {
        alert('Сначала загрузите файл');
        return;
    }
    
    await runAnalysis('video', `/analyze/video/${currentFileId}`);
}

// Запуск аудиоанализа
async function analyzeAudio() {
    if (!currentFileId) {
        alert('Сначала загрузите файл');
        return;
    }
    
    await runAnalysis('audio', `/analyze/audio/${currentFileId}`);
}

// Запуск всех анализов
async function analyzeAll() {
    if (!currentFileId) {
        alert('Сначала загрузите файл');
        return;
    }
    
    // Отключаем кнопки на время анализа
    disableAllButtons(true);
    
    try {
        // Запускаем последовательно
        await runAnalysis('metadata', `/analyze/metadata/${currentFileId}`);
        await runAnalysis('video', `/analyze/video/${currentFileId}`);
        await runAnalysis('audio', `/analyze/audio/${currentFileId}`);
        
        // Показываем итоговый результат
        calculateFinalResult();
        
    } finally {
        disableAllButtons(false);
    }
}

// Выполнение анализа
async function runAnalysis(serviceName, endpoint) {
    const startTime = Date.now();
    
    try {
        // Обновляем статус
        updateServiceDisplay(serviceName, 'processing', '-', 'Анализ выполняется...');
        
        // Отправляем запрос
        const response = await fetch(endpoint);
        const result = await response.json();
        
        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        if (response.ok && result.status === 'success') {
            // Сохраняем результат
            analysisResults[serviceName] = result.result;
            
            // Обновляем отображение
            const probability = result.result.probability_of_ai * 100;
            updateServiceDisplay(
                serviceName,
                'success',
                `${probability.toFixed(1)}%`,
                result.result.explanation || 'Анализ завершен успешно'
            );
            
            document.getElementById(`${serviceName}Time`).textContent = `Выполнено за ${duration} сек`;
            
        } else {
            throw new Error(result.error || 'Неизвестная ошибка');
        }
        
    } catch (error) {
        console.error(`${serviceName} analysis error:`, error);
        
        updateServiceDisplay(
            serviceName,
            'error',
            'Ошибка',
            error.message || 'Не удалось выполнить анализ'
        );
        
        // Устанавливаем вероятность по умолчанию
        analysisResults[serviceName] = { probability_of_ai: 0.0 };
    }
}

// Обновление отображения сервиса
function updateServiceDisplay(serviceName, status, probability, details) {
    const card = document.getElementById(`${serviceName}Card`);
    const statusEl = document.getElementById(`${serviceName}Status`);
    const probEl = document.getElementById(`${serviceName}Probability`);
    const detailsEl = document.getElementById(`${serviceName}Details`);
    
    // Обновляем классы карточки
    card.classList.remove('processing');
    if (status === 'processing') {
        card.classList.add('processing');
    }
    
    // Обновляем статус
    statusEl.textContent = getStatusText(status);
    statusEl.className = `status-badge ${status}`;
    
    // Обновляем вероятность
    probEl.textContent = probability;
    
    // Обновляем детали
    detailsEl.textContent = details;
}

// Получение текста статуса
function getStatusText(status) {
    const statusMap = {
        'waiting': 'Ожидание',
        'processing': 'Выполняется',
        'success': 'Успешно',
        'error': 'Ошибка'
    };
    return statusMap[status] || status;
}

// Вычисление итогового результата
function calculateFinalResult() {
    // Проверяем, что все анализы выполнены
    const allDone = ['metadata', 'video', 'audio'].every(
        service => analysisResults[service]
    );
    
    if (!allDone) return;
    
    // Вычисляем среднюю вероятность
    const probabilities = [
        analysisResults.metadata.probability_of_ai,
        analysisResults.video.probability_of_ai,
        analysisResults.audio.probability_of_ai
    ];
    
    const avgProbability = (Number(probabilities[0]) + Number(probabilities[1]) + Number(probabilities[2])) / 3;
    
    // Определяем вердикт
    const isAI = avgProbability > 0.3;
    const verdict = isAI ? 'NOT AI' : 'AI';
    
    // Обновляем отображение
    document.getElementById('finalMetadataProb').textContent = 
        `${(probabilities[0] * 100).toFixed(1)}%`;
    
    document.getElementById('finalVideoProb').textContent = 
        `${(probabilities[1] * 100).toFixed(1)}%`;
    
    document.getElementById('finalAudioProb').textContent = 
        `${(probabilities[2] * 100).toFixed(1)}%`;
    
    document.getElementById('finalAvgProb').textContent = 
        `${(avgProbability * 100).toFixed(1)}%`;
    
    document.getElementById('confidenceValue').textContent = 
        `${(avgProbability * 100).toFixed(1)}%`;
    
    document.getElementById('verdictText').textContent = verdict;

    const verd = isAI ? 'not-ai' : 'ai';
    document.getElementById('verdictText').className = `verdict-value ${verd}`;
    
    // Показываем итоговый результат
    document.getElementById('finalResult').style.display = 'block';
    document.getElementById('finalResult').scrollIntoView({ behavior: 'smooth' });
}

// Отключение/включение всех кнопок
function disableAllButtons(disabled) {
    const buttons = document.querySelectorAll('.analyze-btn');
    buttons.forEach(btn => {
        btn.disabled = disabled;
        btn.style.opacity = disabled ? 0.6 : 1;
    });
}

// Показать индикатор загрузки
function showUploadProgress() {
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-spinner fa-spin"></i>
        </div>
        <h3>Загрузка...</h3>
        <p>Пожалуйста, подождите</p>
    `;
}

// Скрыть индикатор загрузки
function hideUploadProgress() {
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <h3>Перетащите файл сюда</h3>
        <p>или нажмите для выбора файла</p>
        <p class="file-types">Поддерживаемые форматы: JPG, PNG, MP4, MP3, WAV, AVI</p>
        <input type="file" id="fileInput" accept="image/*,video/*,audio/*">
    `;
    
    // Переназначаем обработчики
    setupFileUpload();
}

// Сбросить весь анализ
function resetAnalysis() {
    if (currentFileId) {
        // Удаляем файл с сервера
        fetch(`/files/${currentFileId}`, { method: 'DELETE' });
    }
    
    // Сбрасываем состояние
    currentFileId = null;
    analysisResults = {
        metadata: null,
        video: null,
        audio: null
    };
    
    // Скрываем все секции
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('analysisControls').style.display = 'none';
    document.getElementById('finalResult').style.display = 'none';
    
    // Сбрасываем отображение
    resetResults();
    
    // Прокручиваем к началу
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Проверка здоровья системы
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const statusEl = document.getElementById('systemStatus');
        const icon = statusEl.querySelector('.fa-circle');
        
        const allHealthy = data.decision_block === 'healthy' && 
                          Object.values(data.services).every(v => v === true);
        
        if (allHealthy) {
            allHealthy = true;
        } else {
            allHealthy = false;
        }
        
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// Форматирование размера файла
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Б';
    const k = 1024;
    const sizes = ['Б', 'КБ', 'МБ', 'ГБ'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function createQR() {
    const response = await fetch('/qr/create', {
        method: 'POST'
    });

    const data = await response.json();
    document.getElementById('qrContainer').innerHTML = `
        <img src="${data.qr_image}" width="250">
    `;

    checkQRUpload(data.session_id);
}

async function checkQRUpload(sessionId) {
    showUploadProgress(); // ? хз надо ли тут вообще это
    const interval = setInterval(async () => {
        const response = await fetch(`/qr/status/${sessionId}`);
        const data = await response.json();

        if (data.uploaded) {
            clearInterval(interval);
            currentFileId = data.file_id;
            showAnalysisControls();
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileName').textContent = data.filename;
            document.getElementById('fileId').textContent = data.file_id;
            document.getElementById('fileSize').textContent = formatFileSize(data.file_size);
            alert("Файл загружен с телефона!");
            hideUploadProgress();
        }
    }, 2000);
}