// On-device symptom inference. Runs the trained SymptomMLP entirely in the
// browser from client-app/model/symptom_mlp_eps5.json, so symptom data never
// has to leave the device. Mirrors models/symptom_mlp.py:
//   Linear(132,256) -> GroupNorm(8) -> ReLU -> Linear(256,128) -> GroupNorm(8)
//   -> ReLU -> Linear(128,64) -> GroupNorm(8) -> ReLU -> Linear(64,41)
// Dropout is a no-op at inference time, so it is skipped.
// Parity with PyTorch is checked by tests/js/model_parity.test.js.

const GROUPS = 8;
const GN_EPS = 1e-5;

function linear(x, W, b) {
    const out = new Array(b.length);
    for (let o = 0; o < b.length; o++) {
        let s = b[o];
        const row = W[o];
        for (let i = 0; i < x.length; i++) s += row[i] * x[i];
        out[o] = s;
    }
    return out;
}

// GroupNorm on a single sample: split channels into GROUPS groups and
// normalize each group over its channels, then apply the per-channel affine.
function groupNorm(x, gamma, beta) {
    const C = x.length;
    const per = C / GROUPS;
    const out = new Array(C);
    for (let g = 0; g < GROUPS; g++) {
        const start = g * per;
        let mean = 0;
        for (let i = start; i < start + per; i++) mean += x[i];
        mean /= per;
        let v = 0;
        for (let i = start; i < start + per; i++) v += (x[i] - mean) ** 2;
        v /= per;
        const inv = 1 / Math.sqrt(v + GN_EPS);
        for (let i = start; i < start + per; i++) {
            out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
    return out;
}

const relu = (x) => x.map((v) => (v > 0 ? v : 0));

function softmax(x) {
    const m = Math.max(...x);
    const e = x.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
}

class LocalSymptomModel {
    constructor(payload) {
        this.vocab = payload.vocab;
        this.classes = payload.classes;
        this.metrics = {
            epsilon: payload.config.epsilon,
            delta: payload.config.delta,
            rounds: payload.config.rounds,
            test_accuracy: payload.test_accuracy,
            test_macro_f1: payload.test_macro_f1,
        };
        const w = payload.weights;
        this.layers = [
            { W: w['net.0.weight'], b: w['net.0.bias'] },
            { gamma: w['net.1.weight'], beta: w['net.1.bias'] },
            { W: w['net.4.weight'], b: w['net.4.bias'] },
            { gamma: w['net.5.weight'], beta: w['net.5.bias'] },
            { W: w['net.8.weight'], b: w['net.8.bias'] },
            { gamma: w['net.9.weight'], beta: w['net.9.bias'] },
            { W: w['net.12.weight'], b: w['net.12.bias'] },
        ];
        this.symptomIndex = new Map(this.vocab.map((s, i) => [s, i]));
    }

    static async load(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`model file not found (${res.status})`);
        return new LocalSymptomModel(await res.json());
    }

    // Returns logits for a binary symptom vector (length = vocab.length).
    logits(features) {
        const [A, N1, B, N2, C, N3, D] = this.layers;
        let h = linear(features, A.W, A.b);
        h = relu(groupNorm(h, N1.gamma, N1.beta));
        h = linear(h, B.W, B.b);
        h = relu(groupNorm(h, N2.gamma, N2.beta));
        h = linear(h, C.W, C.b);
        h = relu(groupNorm(h, N3.gamma, N3.beta));
        return linear(h, D.W, D.b);
    }

    // Input: array of symptom names (as used by the API / checkbox list).
    // Output: same shape as /predict/symptoms -> top_predictions.
    predict(symptoms, topK = 5) {
        const features = new Array(this.vocab.length).fill(0);
        const unknown = [];
        for (const s of symptoms) {
            const idx = this.symptomIndex.get(s);
            if (idx === undefined) unknown.push(s);
            else features[idx] = 1;
        }
        if (unknown.length) throw new Error(`Unrecognized symptom(s): ${unknown.join(', ')}`);

        const probs = softmax(this.logits(features));
        const ranked = probs
            .map((p, i) => ({ disease: this.classes[i], confidence: p, probability: p }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, topK);
        return {
            top_predictions: ranked,
            primary: ranked[0],
            model: { trained: true, on_device: true, ...this.metrics },
        };
    }
}

if (typeof window !== 'undefined') {
    window.LocalSymptomModel = LocalSymptomModel;
    // Resolves to the loaded model, or null if the file can't be fetched.
    // app.js uses the local model when present and falls back to the API.
    window.localModelReady = LocalSymptomModel.load('model/symptom_mlp_eps5.json').catch(() => null);
}
if (typeof module !== 'undefined') module.exports = LocalSymptomModel;
