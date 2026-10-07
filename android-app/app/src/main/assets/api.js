// Android WebView loads these assets from the APK, not from the backend
// server, so this must point at wherever server/api_server.py is actually
// running -- the emulator's host-loopback alias (10.0.2.2) for local dev
// against a backend running on your dev machine, or a real deployed URL
// for a release build. Override by setting window.FEDHEALTH_API_BASE
// before this script loads (e.g. injected by the native WebView host).
const API_BASE = window.FEDHEALTH_API_BASE || 'http://10.0.2.2:8000';

class ApiClient {
    static async fetchWithRetry(url, options = {}, retries = 1) {
        const timeout = 5000;

        for (let i = 0; i <= retries; i++) {
            try {
                const controller = new AbortController();
                const id = setTimeout(() => controller.abort(), timeout);

                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal
                });
                clearTimeout(id);

                if (!response.ok) {
                    let detail = `HTTP error! status: ${response.status}`;
                    try {
                        const body = await response.json();
                        if (body && body.detail) detail = body.detail;
                    } catch (_) { /* body wasn't JSON */ }
                    const httpError = new Error(detail);
                    httpError.status = response.status;
                    throw httpError;
                }
                return await response.json();
            } catch (err) {
                // Don't retry client errors (4xx) -- retrying won't fix a bad request or a 503 "not trained" state
                const isClientError = err.status >= 400 && err.status < 500;
                const isLastAttempt = i === retries || isClientError;
                if (isLastAttempt) {
                    console.error('API request failed:', err);
                    throw new Error(err.name === 'AbortError' ? 'Request timed out' : err.message || 'Network error');
                }
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }
    }

    static async predictSymptoms(symptomsList) {
        const result = await this.fetchWithRetry(`${API_BASE}/predict/symptoms`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: symptomsList })
        });

        if (result && result.top_predictions) {
            result.top_predictions = result.top_predictions.map(p => {
                const val = typeof p.confidence === 'number' ? p.confidence : (typeof p.probability === 'number' ? p.probability : 0);
                return { ...p, confidence: val, probability: val };
            });
        }
        return result;
    }

    static async predictSkin(imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);

        const result = await this.fetchWithRetry(`${API_BASE}/predict/skin`, {
            method: 'POST',
            body: formData
        });

        if (result && result.top_predictions) {
            result.top_predictions = result.top_predictions.map(p => {
                const val = typeof p.confidence === 'number' ? p.confidence : (typeof p.probability === 'number' ? p.probability : 0);
                const name = p.disease || p.condition || p.lesion_type || 'Unknown';
                return { disease: name, condition: name, confidence: val, probability: val };
            });
        }
        return result;
    }

    static async predictRespiratory(audioFile) {
        const formData = new FormData();
        formData.append('file', audioFile);

        const result = await this.fetchWithRetry(`${API_BASE}/predict/respiratory`, {
            method: 'POST',
            body: formData
        });

        if (result && result.top_predictions) {
            result.top_predictions = result.top_predictions.map(p => {
                const val = typeof p.confidence === 'number' ? p.confidence : (typeof p.probability === 'number' ? p.probability : 0);
                const name = p.disease || p.condition || 'Unknown';
                return { disease: name, condition: name, confidence: val, probability: val };
            });
        }
        return result;
    }

    static async getModelStatus() {
        return await this.fetchWithRetry(`${API_BASE}/model/status`);
    }

    static async getPrivacyBudget() {
        return await this.fetchWithRetry(`${API_BASE}/privacy/budget`);
    }
}

window.api = ApiClient;
