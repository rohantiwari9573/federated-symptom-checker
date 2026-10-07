// Parity check: the in-browser forward pass (client-app/model.js) must match
// the PyTorch SymptomMLP that produced client-app/model/*.json.
// Run with:  node tests/js/model_parity.test.js
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const LocalSymptomModel = require('../../client-app/model.js');

const root = path.join(__dirname, '..', '..', 'client-app', 'model');
const payload = JSON.parse(fs.readFileSync(path.join(root, 'symptom_mlp_eps5.json'), 'utf8'));
const parity = JSON.parse(fs.readFileSync(path.join(root, 'parity_test_set.json'), 'utf8'));

const model = new LocalSymptomModel(payload);
const n = parity.X.length;
let argmaxMatches = 0;
let maxLogitDiff = 0;

for (let r = 0; r < n; r++) {
    const jsLogits = model.logits(parity.X[r]);
    const torchLogits = parity.y_torch_logits[r];

    let jsArg = 0, torchArg = 0;
    for (let c = 0; c < jsLogits.length; c++) {
        if (jsLogits[c] > jsLogits[jsArg]) jsArg = c;
        if (torchLogits[c] > torchLogits[torchArg]) torchArg = c;
        maxLogitDiff = Math.max(maxLogitDiff, Math.abs(jsLogits[c] - torchLogits[c]));
    }
    if (jsArg === torchArg) argmaxMatches++;
}

console.log(`rows=${n} argmax_matches=${argmaxMatches} max_logit_abs_diff=${maxLogitDiff.toExponential(3)}`);
assert.strictEqual(argmaxMatches, n, 'argmax disagrees with PyTorch on some rows');
assert.ok(maxLogitDiff < 1e-4, `max logit difference ${maxLogitDiff} exceeds 1e-4`);
console.log('PASS: JS forward pass matches PyTorch.');
