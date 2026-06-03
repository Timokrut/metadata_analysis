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
        updateServiceDisplay(serviceName, 'processing', '-', 'Анализ выполняется...');
        
        const response = await fetch(endpoint);
        const data = await response.json();
        
        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        if (data.status === 'success') {
            // Сохраняем сырой результат
            analysisResults[serviceName] = data.result;
            
            // Извлекаем вероятность AI и объяснение через адаптер
            const { ai_probability, explanation } = extractServiceResult(serviceName, data.result);
            
            const probPercent = ((1 - ai_probability) * 100).toFixed(1);

            updateServiceDisplay(
                serviceName,
                'success',
                `${probPercent}%`,
                explanation || 'Анализ завершён'
            );
            
            document.getElementById(`${serviceName}Time`).textContent = 
                `Выполнено за ${duration} сек`;
        } else {
            throw new Error(data.error || 'Неизвестная ошибка');
        }
    } catch (error) {
        console.error(`${serviceName} analysis error:`, error);
        updateServiceDisplay(serviceName, 'error', 'Ошибка', error.message);
        analysisResults[serviceName] = null;
    }
}

function extractServiceResult(serviceName, result) {
    // Защита от неопределённого или пустого результата
    if (!result) {
        return { ai_probability: 0.0, explanation: 'Нет данных' };
    }

    // Новый формат (метаданные видео) с полями ai_probability, real_probability и т.д.
    if (serviceName === 'metadata') {
        // parse all fields
        const ai_probability = result.ai_probability;
        const ai_software_score = result.ai_software_score;
        const camera_score = result.camera_score;
        const confidence = result.confidence;
        const contributions = result.contributions;
        const gps_score = result.gps_score;
        const known_camera_score = result.known_camera_score;
        const metadata_richness = result.metadata_richness;
        const real_probability  = result.real_probability;
        const statistical_score = result.statistical_score;
        const statistical_score_norm  = result.statistical_score_norm;
        const timestamp_score = result.timestamp_score;
        const verdict = result.verdict;

        const details = [];

        if (result.ai_probability !== undefined) details.push(`📂 AI вероятность: ${(result.ai_probability*100).toFixed(0)}%`);
        if (result.ai_software_score !== undefined) details.push(`🤖 AI ПО: ${(result.ai_software_score*100).toFixed(0)}%`);
        if (result.camera_score !== undefined) details.push(`📷 Камера: ${(result.camera_score*100).toFixed(0)}%`);
        if (result.known_camera_score !== undefined) details.push(`📸 Известная камера: ${(result.known_camera_score*100).toFixed(0)}%`);
        if (result.gps_score !== undefined) details.push(`🌍 GPS: ${(result.gps_score*100).toFixed(0)}%`);
        if (result.confidence !== undefined) details.push(`🔍 Уверенность: ${(result.confidence*100).toFixed(0)}%`);
        if (result.metadata_richness !== undefined) details.push(`📁 Богатство метаданных: ${(result.metadata_richness*100).toFixed(0)}%`);
        if (result.timestamp_score !== undefined) details.push(`⏰ Временная метка: ${(result.timestamp_score*100).toFixed(0)}%`);
        // if (result.contributions !== undefined) details.push(`🤝 Вклад факторов: ${(result.contributions*100).toFixed(0)}%`);
        // if (result.real_probability !== undefined) details.push(`✅ Реальное: ${(result.real_probability*100).toFixed(0)}%`);
        // if (result.statistical_score !== undefined) details.push(`📊 Стат. анализ: ${(result.statistical_score*100).toFixed(0)}%`);
        // if (result.statistical_score_norm !== undefined) details.push(`📈 Норм. стат. анализ: ${(result.statistical_score_norm*100).toFixed(0)}%`);
        const explanation = details.join('<br>') || `Вердикт: ${result.verdict || '—'}`;
        return { ai_probability, explanation };
    }
    
    if (serviceName === 'audio') {
        const result = data.result;   // предположим, что здесь лежат все поля как в вашем HTML

        const details = [];

        // Эмбеддинг статистика
        if (result.embedding_stats) {
            const es = result.embedding_stats;
            details.push('📊 <b>Эмбеддинг статистика</b>');
            if (es.mean !== undefined) details.push(`Среднее: ${es.mean.toFixed(4)}`);
            if (es.std !== undefined) details.push(`Станд. отклонение: ${es.std.toFixed(4)}`);
            if (es.min !== undefined) details.push(`Минимум: ${es.min.toFixed(4)}`);
            if (es.max !== undefined) details.push(`Максимум: ${es.max.toFixed(4)}`);
            if (result.embedding_size) details.push(`Размерность: ${result.embedding_size}`);
        }

        // Акустические признаки
        if (result.acoustic_features) {
            const ac = result.acoustic_features;
            details.push('<br>🎵 <b>Акустические признаки</b>');
            if (ac.spectral_centroid_mean !== undefined) details.push(`Спектр. центроид: ${ac.spectral_centroid_mean.toFixed(0)} Hz`);
            if (ac.rms_mean !== undefined) details.push(`RMS энергия: ${ac.rms_mean.toFixed(4)}`);
            if (ac.zcr_mean !== undefined) details.push(`Zero-crossing rate: ${ac.zcr_mean.toFixed(4)}`);
            if (ac.spectral_entropy !== undefined) details.push(`Спектр. энтропия: ${ac.spectral_entropy.toFixed(2)}`);
        }

        // Спектральные характеристики
        if (result.acoustic_features) {
            const ac = result.acoustic_features;
            details.push('<br>📈 <b>Спектральные характеристики</b>');
            if (ac.spectral_centroid_mean !== undefined) details.push(`Центроид (ср): ${ac.spectral_centroid_mean.toFixed(0)} Hz`);
            if (ac.spectral_centroid_std !== undefined) details.push(`Центроид (стд): ${ac.spectral_centroid_std.toFixed(0)} Hz`);
            if (ac.spectral_bandwidth_mean !== undefined) details.push(`Ширина полосы: ${ac.spectral_bandwidth_mean.toFixed(0)} Hz`);
            if (ac.spectral_rolloff_mean !== undefined) details.push(`Роллофф: ${ac.spectral_rolloff_mean.toFixed(0)} Hz`);
            if (ac.spectral_flatness_mean !== undefined) details.push(`Плоскостность: ${ac.spectral_flatness_mean.toFixed(4)}`);
        }

        // Временные характеристики
        if (result.acoustic_features) {
            const ac = result.acoustic_features;
            details.push('<br>⏱️ <b>Временные характеристики</b>');
            if (ac.amplitude_max !== undefined) details.push(`Амплитуда (макс): ${ac.amplitude_max.toFixed(4)}`);
            if (ac.amplitude_mean !== undefined) details.push(`Амплитуда (ср): ${ac.amplitude_mean.toFixed(4)}`);
            if (ac.rms_std !== undefined) details.push(`RMS (стд): ${ac.rms_std.toFixed(4)}`);
            if (ac.zcr_std !== undefined) details.push(`ZCR (стд): ${ac.zcr_std.toFixed(4)}`);
        }

        const explanation = details.join('<br>') || `Вердикт: ${result.verdict || '—'}`;
        return {
            ai_probability: result.score,
            explanation
        };
    }

        if (serviceName === 'video') {
            return {
                ai_probability: result.probability_of_ai,
                explanation: result.explanation || ''
            };
        }
        
        // Заглушка, если формат неизвестен
        return { ai_probability: 0.0, explanation: 'Неизвестный формат результата' };
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
    detailsEl.innerHTML = details;
    // detailsEl.style.whiteSpace = 'pre-line';   
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
    const services = ['metadata', 'video', 'audio'];
    const allDone = services.every(s => analysisResults[s] !== null && analysisResults[s] !== undefined);
    if (!allDone) return;
    
    // Извлекаем AI-вероятности
    const probs = services.map(s => 1 - extractServiceResult(s, analysisResults[s]).ai_probability);
    
    // Средняя вероятность AI
    const avgAI = probs.reduce((a, b) => a + b, 0) / probs.length;
    
    // Вердикт: если средняя AI-вероятность >= 0.5, считаем AI
    const isAI = avgAI > 0.5;
    const verdict = isAI ? 'NOT AI' : 'AI';
    
    // Заполняем интерфейс
    document.getElementById('finalMetadataProb').textContent = `${(probs[0] * 100).toFixed(1)}%`;
    document.getElementById('finalVideoProb').textContent = `${(probs[1] * 100).toFixed(1)}%`;
    document.getElementById('finalAudioProb').textContent = `${(probs[2] * 100).toFixed(1)}%`;
    document.getElementById('finalAvgProb').textContent = `${(avgAI * 100).toFixed(1)}%`;
    document.getElementById('confidenceValue').textContent = `${(avgAI * 100).toFixed(1)}%`;
    
    const verdictEl = document.getElementById('verdictText');
    verdictEl.textContent = verdict;
    verdictEl.className = `verdict-value ${isAI ? 'ai' : 'real'}`;
    
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