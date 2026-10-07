document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSymptomChecker();
    initSkinAnalysis();
    initRespiratoryAnalysis();
    initPrivacyDashboard();
    initResultsPanel();
});

// --- Tab Navigation Logic ---
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-pane');

    function switchTab(targetId) {
        history.replaceState(null, null, `#${targetId}`);

        tabs.forEach(t => {
            t.classList.remove('active');
            if (t.dataset.target === targetId) {
                t.classList.add('active');
            }
        });

        panes.forEach(p => {
            if (p.id === targetId) {
                p.classList.remove('hidden');
                void p.offsetWidth;
                p.classList.add('fade-in');
            } else {
                p.classList.add('hidden');
                p.classList.remove('fade-in');
            }
        });
        
        hideResults();
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.target));
    });

    const hash = window.location.hash.substring(1);
    const validTabs = ['symptoms', 'skin', 'respiratory'];
    if (validTabs.includes(hash)) {
        switchTab(hash);
    }
}

// --- Symptom Checker Logic ---
async function initSymptomChecker() {
    let symptomsList = [];

    const formatName = (str) => str.trim().replace(/\.\d+$/, '').replace(/\s+/g, '_')
        .split('_').filter(Boolean).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

    const container = document.getElementById('symptomList');
    const searchInput = document.getElementById('symptomSearch');
    const tagsContainer = document.getElementById('selectedSymptoms');
    const analyzeBtn = document.getElementById('analyzeSymptomsBtn');

    let selectedSymptoms = new Set();

    try {
        const vocab = await window.api.getSymptomVocabulary();
        symptomsList = vocab.symptoms || [];
        if (!vocab.trained || symptomsList.length === 0) {
            console.warn("Symptom model not trained on server yet. Using reference symptom set.");
            symptomsList = getDefaultSymptoms();
        }
    } catch (error) {
        console.warn('Could not reach API server for dynamic symptoms, loading default vocabulary:', error);
        symptomsList = getDefaultSymptoms();
    }

    function getDefaultSymptoms() {
        return [
            "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_typhos", "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", "blood_in_sputum", "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
        ];
    }

    function renderList(filter = '') {
        container.innerHTML = '';
        const filtered = symptomsList.filter(s => formatName(s).toLowerCase().includes(filter.toLowerCase()));
        
        filtered.forEach(sym => {
            const item = document.createElement('div');
            const isChecked = selectedSymptoms.has(sym);
            item.className = `symptom-item-neo ${isChecked ? 'checked' : ''}`;
            item.innerHTML = `
                <div class="checkbox-box">${isChecked ? '✓' : ''}</div>
                <span class="symptom-label">${formatName(sym)}</span>
            `;
            
            item.addEventListener('click', () => {
                if (selectedSymptoms.has(sym)) {
                    selectedSymptoms.delete(sym);
                    item.classList.remove('checked');
                    item.querySelector('.checkbox-box').textContent = '';
                } else {
                    selectedSymptoms.add(sym);
                    item.classList.add('checked');
                    item.querySelector('.checkbox-box').textContent = '✓';
                }
                renderTags();
                updateBtnState();
            });
            container.appendChild(item);
        });
    }

    function renderTags() {
        tagsContainer.innerHTML = '';
        selectedSymptoms.forEach(sym => {
            const tag = document.createElement('div');
            tag.className = 'tag-selected';
            tag.innerHTML = `
                ${formatName(sym)}
                <span class="tag-remove" data-val="${sym}">×</span>
            `;
            tag.querySelector('.tag-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                selectedSymptoms.delete(sym);
                renderTags();
                renderList(searchInput.value);
                updateBtnState();
            });
            tagsContainer.appendChild(tag);
        });
    }

    function updateBtnState() {
        analyzeBtn.disabled = selectedSymptoms.size === 0;
    }

    searchInput.addEventListener('input', (e) => renderList(e.target.value));

    analyzeBtn.addEventListener('click', async () => {
        setLoading(analyzeBtn, true);
        try {
            // Prefer on-device inference (no symptom data leaves the browser).
            // Fall back to the API only if the bundled model file is missing.
            const local = window.localModelReady ? await window.localModelReady : null;
            const results = local
                ? local.predict(Array.from(selectedSymptoms))
                : await window.api.predictSymptoms(Array.from(selectedSymptoms));
            const primary = results.primary || { disease: results.disease, confidence: results.confidence };
            displayResults(primary.disease, primary.confidence, results.top_predictions, results);
        } catch (error) {
            alert((error.message === 'Network error' || error.message === 'Request timed out') ? 'Could not reach the analysis server. This demo needs a backend running -- see the project README for setup.' : 'Analysis failed: ' + error.message);
        } finally {
            setLoading(analyzeBtn, false);
        }
    });

    renderList();
    updateBtnState();
}

// --- Skin Analysis Logic ---
function initSkinAnalysis() {
    const dropZone = document.getElementById('skinDropZone');
    const fileInput = document.getElementById('skinFileInput');
    const content = document.getElementById('skinUploadContent');
    const previewContainer = document.getElementById('skinPreviewContainer');
    const previewImg = document.getElementById('skinPreview');
    const removeBtn = document.getElementById('removeSkinFile');
    const analyzeBtn = document.getElementById('analyzeSkinBtn');
    
    let currentFile = null;

    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#removeSkinFile')) return;
        fileInput.click();
    });
    
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
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
            alert('Please select an image file');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            content.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewContainer.classList.add('hidden');
        content.classList.remove('hidden');
        analyzeBtn.disabled = true;
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        setLoading(analyzeBtn, true);
        try {
            const results = await window.api.predictSkin(currentFile);
            const primaryName = results.lesion_type || results.disease || results.condition;
            displayResults(primaryName, results.confidence, results.top_predictions, results);
        } catch (error) {
            alert((error.message === 'Network error' || error.message === 'Request timed out') ? 'Could not reach the analysis server. This demo needs a backend running -- see the project README for setup.' : 'Image analysis failed: ' + error.message);
        } finally {
            setLoading(analyzeBtn, false);
        }
    });
}

// --- Audio helpers ---
async function encodeBlobAsWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    audioCtx.close();

    const numChannels = decoded.numberOfChannels;
    const sampleRate = decoded.sampleRate;
    const numFrames = decoded.length;

    // Interleave channels
    const interleaved = new Float32Array(numFrames * numChannels);
    for (let ch = 0; ch < numChannels; ch++) {
        const channelData = decoded.getChannelData(ch);
        for (let i = 0; i < numFrames; i++) {
            interleaved[i * numChannels + ch] = channelData[i];
        }
    }

    const bytesPerSample = 2; // 16-bit PCM
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = interleaved.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);       // PCM chunk size
    view.setUint16(20, 1, true);        // audio format = PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < interleaved.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, interleaved[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// --- Respiratory Analysis Logic ---
function initRespiratoryAnalysis() {
    const recordBtn = document.getElementById('recordBtn');
    const timerEl = document.getElementById('recordingTimer');
    const statusEl = document.getElementById('recordingStatus');
    const dropZone = document.getElementById('audioDropZone');
    const fileInput = document.getElementById('audioFileInput');
    const playerContainer = document.getElementById('audioPlayerContainer');
    const player = document.getElementById('audioPlayback');
    const removeBtn = document.getElementById('removeAudioFile');
    const analyzeBtn = document.getElementById('analyzeAudioBtn');

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let currentFile = null;

    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#removeAudioFile')) return;
        fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleAudioFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleAudioFile(e.target.files[0]);
        }
    });

    function handleAudioFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }
        currentFile = file;
        player.src = URL.createObjectURL(file);
        playerContainer.classList.remove('hidden');
        dropZone.classList.add('hidden');
        analyzeBtn.disabled = false;
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        player.src = '';
        playerContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        analyzeBtn.disabled = true;
    });

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                    try {
                        const wavBlob = await encodeBlobAsWav(rawBlob);
                        currentFile = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
                        player.src = URL.createObjectURL(wavBlob);
                        playerContainer.classList.remove('hidden');
                        dropZone.classList.add('hidden');
                        analyzeBtn.disabled = false;
                        statusEl.textContent = 'Recording captured and converted to WAV.';
                    } catch (err) {
                        console.error('Failed to encode recording as WAV:', err);
                        alert('Could not transcode audio to 16-bit PCM WAV. Try uploading an audio file instead.');
                        statusEl.textContent = 'Audio processing error.';
                    }
                };

                mediaRecorder.start();
                isRecording = true;
                recordBtn.classList.add('recording');
                statusEl.textContent = 'Recording in progress... Ticker active.';
                startTimer();
            } catch (err) {
                alert('Microphone access denied or unavailable: ' + err.message);
            }
        } else {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
            }
            isRecording = false;
            recordBtn.classList.remove('recording');
            statusEl.textContent = 'Processing recorded audio...';
            stopTimer();
        }
    });

    function startTimer() {
        let sec = 0;
        timerInterval = setInterval(() => {
            sec++;
            const m = Math.floor(sec / 60).toString().padStart(2, '0');
            const s = (sec % 60).toString().padStart(2, '0');
            timerEl.textContent = `${m}:${s}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        setLoading(analyzeBtn, true);
        try {
            const results = await window.api.predictRespiratory(currentFile);
            const primaryName = results.condition || results.disease;
            displayResults(primaryName, results.confidence, results.top_predictions, results);
        } catch (error) {
            alert((error.message === 'Network error' || error.message === 'Request timed out') ? 'Could not reach the analysis server. This demo needs a backend running -- see the project README for setup.' : 'Respiratory sound analysis failed: ' + error.message);
        } finally {
            setLoading(analyzeBtn, false);
        }
    });
}

// --- Privacy Dashboard Telemetry ---
async function initPrivacyDashboard() {
    async function updateData() {
        try {
            // The bundled on-device model is the source of truth for the
            // dashboard, so ε, round and accuracy describe the weights in use.
            const local = window.localModelReady ? await window.localModelReady : null;
            if (local) {
                const m = local.metrics;
                document.getElementById('epsilonValue').textContent = `ε = ${m.epsilon.toFixed(2)}`;
                document.getElementById('epsilonFill').style.width = `${Math.min((m.epsilon / 10.0) * 100, 100)}%`;
                const note = document.getElementById('epsilonNote');
                if (note) note.textContent = `DP-SGD, Opacus-calibrated (δ = ${m.delta})`;
                document.getElementById('currentRound').textContent = m.rounds;
                document.getElementById('totalRounds').textContent = m.rounds;
                document.getElementById('globalAccuracy').textContent = (m.test_accuracy * 100).toFixed(1);
                return;
            }

            const [status, privacy] = await Promise.all([
                window.api.getModelStatus(),
                window.api.getPrivacyBudget()
            ]);

            const epsVal = privacy.epsilon != null ? privacy.epsilon : '2.00';
            document.getElementById('epsilonValue').textContent = `ε = ${epsVal}`;
            const epsNum = parseFloat(epsVal) || 2.0;
            const epsPct = Math.min((epsNum / 10.0) * 100, 100);
            document.getElementById('epsilonFill').style.width = `${epsPct}%`;

            const epsilonNote = document.getElementById('epsilonNote');
            if (epsilonNote) {
                epsilonNote.textContent = privacy.dp_applied_to_deployed_model === false
                    ? 'Reference benchmark only (FedAvg without DP deployed)'
                    : 'OPACUS DP-SGD (δ = 1e-5)';
            }

            document.getElementById('currentRound').textContent = status.current_round != null ? status.current_round : '--';
            document.getElementById('totalRounds').textContent = status.total_rounds != null ? status.total_rounds : '--';

            const accEl = document.getElementById('globalAccuracy');
            if (status.global_accuracy != null) {
                const accVal = typeof status.global_accuracy === 'number' && status.global_accuracy <= 1
                    ? (status.global_accuracy * 100).toFixed(1)
                    : status.global_accuracy;
                accEl.textContent = `${accVal}`;
            } else {
                accEl.textContent = '--';
            }
        } catch (err) {
            console.warn('Privacy telemetry polling fallback:', err);
            const epsNote = document.getElementById('epsilonNote');
            if (epsNote) epsNote.textContent = 'API Server Offline';
        }
    }

    await updateData();
    setInterval(updateData, 30000);
}

// --- Diagnostic Results Panel ---
function initResultsPanel() {
    document.getElementById('closeResults').addEventListener('click', hideResults);
}

function hideResults() {
    const panel = document.getElementById('resultsPanel');
    panel.classList.add('hidden');
}

function displayResults(primaryName, primaryConf, topPredictions, fullResult) {
    const panel = document.getElementById('resultsPanel');
    
    // Demo notice for untrained heads
    const demoNotice = document.getElementById('demoNotice');
    const demoNoticeText = document.getElementById('demoNoticeText');
    if (demoNotice && demoNoticeText) {
        if (fullResult && fullResult.trained === false) {
            demoNoticeText.textContent = fullResult.notice || 'This model has not been trained on real data yet -- result is illustrative only.';
            demoNotice.classList.remove('hidden');
        } else {
            demoNotice.classList.add('hidden');
        }
    }

    const validPrimaryConf = typeof primaryConf === 'number' && !isNaN(primaryConf) ? primaryConf : 0;
    const confPercent = Math.round(validPrimaryConf * 100);
    
    document.getElementById('primaryDiagnosis').textContent = primaryName || 'Unknown';
    
    animateValue('primaryConfidence', 0, confPercent, 1000);
    
    const arc = document.getElementById('primaryConfidenceArc');
    let color = '#FF3B30'; // High-voltage Coral Red
    if (confPercent >= 80) color = '#10B981'; // Green
    else if (confPercent >= 50) color = '#F59E0B'; // Yellow
    
    arc.style.stroke = color;
    
    setTimeout(() => {
        arc.style.strokeDasharray = `${confPercent}, 100`;
    }, 100);

    const list = document.getElementById('predictionsList');
    list.innerHTML = '';
    
    (topPredictions || []).forEach((pred, index) => {
        const name = pred.disease || pred.condition || 'Unknown';
        const rawProb = typeof pred.probability === 'number' ? pred.probability : (typeof pred.confidence === 'number' ? pred.confidence : 0);
        const conf = isNaN(rawProb) || !isFinite(rawProb) ? 0 : Math.round(rawProb * 100);
        
        let barColor = 'var(--accent-coral)';
        if (conf < 30) barColor = 'var(--text-muted)';
        else if (conf < 60) barColor = 'var(--accent-yellow)';

        const item = document.createElement('div');
        item.className = 'prediction-item-neo';
        item.innerHTML = `
            <div class="pred-name-neo" title="${name}">${name}</div>
            <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="background: ${barColor};" data-width="${conf}%"></div>
            </div>
            <div class="pred-val-neo">${conf}%</div>
        `;
        list.appendChild(item);
        
        setTimeout(() => {
            const bar = item.querySelector('.pred-bar-fill');
            if (bar) bar.style.width = bar.dataset.width;
        }, 100 + (index * 100));
    });

    panel.classList.remove('hidden');
    void panel.offsetWidth;
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function setLoading(btn, isLoading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    if (isLoading) {
        btn.disabled = true;
        if (text) text.classList.add('hidden');
        if (loader) loader.classList.remove('hidden');
    } else {
        btn.disabled = false;
        if (text) text.classList.remove('hidden');
        if (loader) loader.classList.add('hidden');
    }
}
