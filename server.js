const express = require('express');
const axios = require('axios');
const app = express();
const port = process.env.PORT || 3000;

// Configuration
const CONFIG = {
  API_URL: "https://sun-predict-5ghi.onrender.com/api/taixiu/sunwin",
  HISTORY_KEY: "sun_predict_history_v1",
  PATTERN_MEM_KEY: "sun_predict_pattern_mem_v1",
  ERROR_MEM_KEY: "sun_predict_error_mem_v1",
  AUTO_REFRESH_DEFAULT: 0,
  MAX_HISTORY_STORE: 2000,
  MARKOV_ORDER: 3,
  RUN_WINDOW_SHORT: 6,
  RUN_WINDOW_LONG: 20,
  BASE_CONFIDENCE: 0.5,
  MODELS: ['markov', 'run_length', 'momentum', 'pattern'],
};

// In-memory storage (replace with a database in production)
let PATTERN_MEMORY = {};
let ERROR_MEMORY = {};
let history = [];

// Utility Functions
function nowStr() { return (new Date()).toISOString(); }
function deepCopy(x) { return JSON.parse(JSON.stringify(x)); }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function last(arr, n = 1) { return arr.slice(Math.max(arr.length - n, 0)); }
function seqFromHistory(history) {
  return history.map(h => h.Ket_qua === 'Tài' || h.Ket_qua === 'Tai' || h.Ket_qua === 'T' ? 'T' : 'X');
}

// Markov Model
class MarkovModel {
  constructor(order = CONFIG.MARKOV_ORDER) {
    this.order = order;
    this.table = {};
  }
  train(seq) {
    this.table = {};
    for (let i = 0; i + this.order < seq.length; i++) {
      const ctx = seq.slice(i, i + this.order).join('');
      const next = seq[i + this.order];
      if (!this.table[ctx]) this.table[ctx] = { T: 0, X: 0 };
      this.table[ctx][next] += 1;
    }
  }
  predictProba(seq) {
    if (seq.length < this.order) {
      const c = counts(seq);
      const tot = (c.T + c.X) || 1;
      return { T: c.T / tot, X: c.X / tot };
    }
    const ctx = seq.slice(seq.length - this.order).join('');
    const entry = this.table[ctx];
    if (!entry) {
      if (this.order > 1) {
        const sub = new MarkovModel(this.order - 1);
        sub.table = this.table;
        return sub.predictProba(seq);
      } else {
        const c = counts(seq);
        const tot = (c.T + c.X) || 1;
        return { T: c.T / tot, X: c.X / tot };
      }
    }
    const tot = entry.T + entry.X || 1;
    return { T: entry.T / tot, X: entry.X / tot };
  }
}

function computeRunLength(seq) {
  if (!seq.length) return { value: null, run: 0 };
  let lastVal = seq[seq.length - 1];
  let run = 1;
  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === lastVal) run++; else break;
  }
  return { value: lastVal, run };
}

function counts(seq) {
  const c = { T: 0, X: 0 };
  seq.forEach(s => { if (s === 'T') c.T++; else c.X++; });
  return c;
}

// Run Length Model
class RunLengthModel {
  predictProba(seq) {
    if (!seq.length) return { T: 0.5, X: 0.5 };
    const { value, run } = computeRunLength(seq);
    const shortThreshold = CONFIG.RUN_WINDOW_SHORT;
    const longThreshold = CONFIG.RUN_WINDOW_LONG;
    let contProb = 0.6 * Math.exp(-run / shortThreshold) + 0.3 * Math.exp(-run / longThreshold);
    contProb = clamp(contProb, 0.05, 0.95);
    const res = { T: 0.5, X: 0.5 };
    if (value === 'T') { res.T = contProb; res.X = 1 - contProb; }
    else { res.X = contProb; res.T = 1 - contProb; }
    return res;
  }
}

// Momentum Model
class MomentumModel {
  predictProba(seq) {
    const nShort = 5, nMid = 15;
    const s1 = last(seq, nShort);
    const s2 = last(seq, nMid);
    const c1 = counts(s1), c2 = counts(s2);
    const scoreShort = (c1.T - c1.X) / (nShort || 1);
    const scoreMid = (c2.T - c2.X) / (nMid || 1);
    let momentum = 0.7 * scoreShort + 0.3 * scoreMid;
    const shift = clamp(momentum * 0.4, -0.4, 0.4);
    const pT = clamp(0.5 + shift, 0.02, 0.98);
    return { T: pT, X: 1 - pT };
  }
}

// Pattern Model
class PatternModel {
  detectPattern(seq) {
    if (seq.length >= 6) {
      const tail = last(seq, 6).join('');
      if (/^([TX])([TX])\1\2\1\2$/.test(tail)) return { type: 'zigzag', strength: 0.9 };
    }
    const { value, run } = computeRunLength(seq);
    if (run >= 4) return { type: 'streak', strength: clamp((run - 3) / 10, 0.2, 0.9) };
    if (seq.length >= 8) {
      const tail = last(seq, 8).join('');
      if (/^(T{2}X{2})+|^(X{2}T{2})+/.test(tail)) return { type: 'twin', strength: 0.85 };
    }
    return { type: 'none', strength: 0.0 };
  }
  predictProba(seq) {
    const p = { T: 0.5, X: 0.5 };
    const detected = this.detectPattern(seq);
    if (detected.type === 'zigzag') {
      const lastVal = seq[seq.length - 1];
      p[lastVal === 'T' ? 'X' : 'T'] = 0.6 * detected.strength + 0.4;
      p[lastVal] = 1 - p[lastVal === 'T' ? 'X' : 'T'];
    } else if (detected.type === 'streak') {
      const lastVal = seq[seq.length - 1];
      p[lastVal] = 0.55 * detected.strength + 0.45;
      p[lastVal === 'T' ? 'X' : 'T'] = 1 - p[lastVal];
    } else if (detected.type === 'twin') {
      const lastVal = seq[seq.length - 1];
      p[lastVal === 'T' ? 'X' : 'T'] = 0.62 * detected.strength + 0.38;
      p[lastVal] = 1 - p[lastVal === 'T' ? 'X' : 'T'];
    }
    return p;
  }
}

// Ensemble Model
class Ensemble {
  constructor() {
    this.weights = {};
    CONFIG.MODELS.forEach(m => this.weights[m] = 1 / CONFIG.MODELS.length);
    this.perfEMA = {};
    CONFIG.MODELS.forEach(m => this.perfEMA[m] = 0.5);
    this.models = {
      markov: new MarkovModel(CONFIG.MARKOV_ORDER),
      run_length: new RunLengthModel(),
      momentum: new MomentumModel(),
      pattern: new PatternModel(),
    };
  }
  trainAll(seq) {
    this.models.markov.train(seq);
  }
  predictProba(seq) {
    const modelProbas = {};
    CONFIG.MODELS.forEach(m => { modelProbas[m] = this.models[m].predictProba(seq); });
    const mix = { T: 0, X: 0 };
    CONFIG.MODELS.forEach(m => {
      const w = this.weights[m] || 0;
      mix.T += w * modelProbas[m].T;
      mix.X += w * modelProbas[m].X;
    });
    const tot = mix.T + mix.X || 1;
    mix.T /= tot; mix.X /= tot;
    return { distribution: mix, modelProbas, weights: { ...this.weights } };
  }
  updateWeights(seqBefore, actual) {
    CONFIG.MODELS.forEach(m => {
      const p = this.models[m].predictProba(seqBefore)[actual];
      const score = clamp(p, 0.001, 0.999);
      const old = this.perfEMA[m] || 0.5;
      const alpha = 0.08;
      this.perfEMA[m] = old * (1 - alpha) + alpha * score;
    });
    const raw = {}; let sumRaw = 0;
    CONFIG.MODELS.forEach(m => { raw[m] = Math.pow(this.perfEMA[m], 3); sumRaw += raw[m]; });
    const newWeights = {};
    CONFIG.MODELS.forEach(m => {
      const target = sumRaw ? raw[m] / sumRaw : 1 / CONFIG.MODELS.length;
      newWeights[m] = clamp(this.weights[m] * (1 - 0.05) + target * 0.05, 0.0001, 0.9999);
    });
    const sumNew = Object.values(newWeights).reduce((a, b) => a + b, 0) || 1;
    CONFIG.MODELS.forEach(m => this.weights[m] = newWeights[m] / sumNew);
  }
}

// Manual Patterns
const MANUAL_PATTERNS = [
  // ... (same as provided in the original code, omitted for brevity)
];

// Match Manual Pattern
function matchManualPattern(totals) {
  const maxCheckLen = 6;
  for (let pat of MANUAL_PATTERNS) {
    const p = pat.pair;
    if (p.length > totals.length) continue;
    let match = true;
    for (let i = 0; i < p.length; i++) {
      if (totals[totals.length - p.length + i] !== p[i]) {
        match = false; break;
      }
    }
    if (match) return { pred: pat.pred, note: pat.note, source: 'manual' };
  }
  return null;
}

// du_doan_js Function
function du_doan_js(data_kq, dem_sai, pattern_sai, xx, diem_lich_su, data_store) {
  try {
    let xx_list = [];
    if (typeof xx === 'string') xx_list = xx.split('-').map(s => s.trim());
    else if (Array.isArray(xx)) xx_list = xx.map(x => String(x));
    const tong = xx_list.reduce((s, x) => s + parseInt(x || 0), 0);
    data_kq = data_kq.map(x => x === 'Tài' || x === 'T' ? 'T' : (x === 'X' || x === 'Xỉu' ? 'X' : (x === 'Xiu' ? 'X' : x)));
    data_kq = data_kq.slice(-100);
    const cuoi = data_kq.length ? (data_kq[data_kq.length - 1]) : null;
    const pattern = data_kq.map(x => x === 'T' ? 'T' : 'X').join('');
    let matched_pattern = null, matched_confidence = 0, matched_pred = null;
    for (let pat in PATTERN_MEMORY) {
      if (pattern.endsWith(pat)) {
        const stats = PATTERN_MEMORY[pat];
        const count = stats.count || 0;
        const correct = stats.correct || 0;
        const confidence = count > 0 ? correct / count : 0;
        if (confidence > matched_confidence && count >= 3 && confidence >= 0.6) {
          matched_confidence = confidence;
          matched_pattern = pat;
          matched_pred = stats.next_pred;
        }
      }
    }
    if (matched_pattern && matched_pred) {
      const score = 90 + Math.floor(matched_confidence * 10);
      return { pred: matched_pred === 'T' ? 'T' : 'X', score, reason: `Dự theo mẫu đã học '${matched_pattern}' tin cậy ${matched_confidence.toFixed(2)}` };
    }
    if (data_kq.length >= 3) {
      const last3 = data_kq.slice(-3).join(',');
      if (ERROR_MEMORY[last3] && ERROR_MEMORY[last3] >= 2) {
        const du = cuoi === 'T' ? 'X' : 'T';
        return { pred: du, score: 89, reason: `AI tự học lỗi: mẫu ${last3} gây sai nhiều → đảo` };
      }
    }
    if (dem_sai >= 4) {
      const du = cuoi === 'T' ? 'X' : 'T';
      return { pred: du, score: 87, reason: `Sai liên tiếp ${dem_sai} → đổi` };
    }
    if (data_kq.length >= 5) {
      const tail5 = data_kq.slice(-5);
      const countT = tail5.filter(x => 'T' === x).length;
      const countX = tail5.filter(x => 'X' === x).length;
      if (countT === countX && data_kq[data_kq.length - 1] !== data_kq[data_kq.length - 2]) {
        const du = cuoi === 'T' ? 'X' : 'T';
        return { pred: du, score: 88, reason: 'Phát hiện dấu hiệu đổi cầu → đổi hướng' };
      }
    }
    if (data_kq.length < 1) {
      if (tong >= 16) return { pred: 'T', score: 98, reason: `Tay đầu tổng ${tong} >=16 → Tài` };
      if (tong <= 6) return { pred: 'X', score: 98, reason: `Tay đầu tổng ${tong} <=6 → Xỉu` };
      return { pred: tong >= 11 ? 'T' : 'X', score: 75, reason: `Tay đầu → Dựa tổng ${tong}` };
    }
    if (data_kq.length == 1) {
      if (tong >= 16) return { pred: 'T', score: 98, reason: `Tay 2 tổng ${tong} >=16 → Tài` };
      if (tong <= 6) return { pred: 'X', score: 98, reason: `Tay 2 tổng ${tong} <=6 → Xỉu` };
      const du = cuoi === 'T' ? 'X' : 'T';
      return { pred: du, score: 80, reason: `Tay 2 → dự đoán ngược (${cuoi})` };
    }
    const ben = (() => {
      if (!data_kq.length) return 0;
      const lastVal = data_kq[data_kq.length - 1];
      let run = 1;
      for (let i = data_kq.length - 2; i >= 0; i--) {
        if (data_kq[i] === lastVal) run++; else break;
      }
      return run;
    })();
    const countsObj = { T: data_kq.filter(x => 'T' === x).length, X: data_kq.filter(x => 'X' === x).length };
    const chenh = Math.abs(countsObj.T - countsObj.X);
    diem_lich_su = diem_lich_su || [];
    diem_lich_su.push(tong);
    if (diem_lich_su.length > 6) diem_lich_su.shift();
    if (pattern.length >= 9) {
      for (let i = 4; i <= 6; i++) {
        if (pattern.length >= i * 2) {
          const sub1 = pattern.slice(-i * 2, -i);
          const sub2 = pattern.slice(-i);
          if (sub1 === 'T'.repeat(i) && sub2 === 'X'.repeat(i)) return { pred: 'X', score: 90, reason: `Phát hiện cầu bệt-bệt ${sub1 + sub2}` };
          if (sub1 === 'X'.repeat(i) && sub2 === 'T'.repeat(i)) return { pred: 'T', score: 90, reason: `Phát hiện cầu bệt-bệt ${sub1 + sub2}` };
        }
      }
    }
    if (diem_lich_su.length >= 3 && (new Set(diem_lich_su.slice(-3))).size === 1) {
      return { pred: (tong % 2 === 1) ? 'T' : 'X', score: 96, reason: `3 lần lặp điểm: ${tong}` };
    }
    if (diem_lich_su.length >= 2 && diem_lich_su[diem_lich_su.length - 1] === diem_lich_su[diem_lich_su.length - 2]) {
      return { pred: (tong % 2 === 0) ? 'T' : 'X', score: 94, reason: `Kép điểm: ${tong}` };
    }
    if (xx_list.length === 3 && xx_list[0] === xx_list[1] && xx_list[1] === xx_list[2]) {
      const so = xx_list[0];
      if (['1', '2', '4'].includes(so)) return { pred: 'X', score: 97, reason: `3 xúc xắc ${so} → Xỉu` };
      if (['3', '5'].includes(so)) return { pred: 'T', score: 97, reason: `3 xúc xắc ${so} → Tài` };
      if (so === '6' && ben >= 3) return { pred: 'T', score: 97, reason: '3 xúc xắc 6 + bệt → Tài' };
    }
    if (ben >= 3) {
      if (cuoi === 'T') {
        if (ben >= 5 && !xx_list.includes('3')) {
          if (!data_store.da_be_tai) { data_store.da_be_tai = true; return { pred: 'X', score: 80, reason: '⚠️ Bệt Tài ≥5 chưa có xx3 → Bẻ thử' }; }
          else return { pred: 'T', score: 90, reason: 'Ôm tiếp bệt Tài chờ xx3' };
        } else if (xx_list.includes('3')) {
          data_store.da_be_tai = false;
          return { pred: 'X', score: 95, reason: 'Bệt Tài + Xí ngầu 3 → Bẻ' };
        }
      } else {
        if (ben >= 5 && !xx_list.includes('5')) {
          if (!data_store.da_be_xiu) { data_store.da_be_xiu = true; return { pred: 'T', score: 80, reason: '⚠️ Bệt Xỉu ≥5 chưa có xx5 → Bẻ thử' }; }
          else return { pred: 'X', score: 90, reason: 'Ôm tiếp bệt Xỉu chờ xx5' };
        } else if (xx_list.includes('5')) {
          data_store.da_be_xiu = false;
          return { pred: 'T', score: 95, reason: 'Bệt Xỉu + Xí ngầu 5 → Bẻ' };
        }
      }
      return { pred: cuoi, score: 93, reason: `Bệt ${cuoi} (${ben} tay)` };
    }
    const ends = (pats) => pats.some(p => pattern.endsWith(p));
    const cau_mau = {
      "1-1": ["TXTX", "XTXT", "TXTXT", "XTXTX"],
      "2-2": ["TTXXTT", "XXTTXX"],
      "3-3": ["TTTXXX", "XXXTTT"],
      "1-2-3": ["TXXTTT", "XTTXXX"],
      "3-2-1": ["TTTXXT", "XXXTTX"],
      "1-2-1": ["TXXT", "XTTX"],
      "2-1-1-2": ["TTXTXX", "XXTXTT"],
      "2-1-2": ["TTXTT", "XXTXX"],
      "3-1-3": ["TTTXTTT", "XXXTXXX"],
      "1-2": ["TXX", "XTT"],
      "2-1": ["TTX", "XXT"],
      "1-3-2": ["TXXXTT", "XTTTXX"],
      "1-2-4": ["TXXTTTT", "XTTXXXX"],
      "1-5-3": ["TXXXXXTTT", "XTTTTXXX"],
      "7-4-2": ["TTTTTTTXXXXTT", "XXXXXXXTTTTXX"],
      "4-2-1-3": ["TTTTXXTXXX", "XXXXTTXTTT"],
      "1-4-2": ["TXXXXTT", "XTTTTXX"],
      "5-1-3": ["TTTTXTTT", "XXXXXTXXX"],
    };
    for (let loai in { "1-1": 1 }) {
      for (let mau of cau_mau["1-1"]) {
        if (pattern.endsWith(mau)) {
          const length_cau = mau.length;
          const current_len = data_kq.length;
          if (length_cau == 4) {
            if (current_len == 5) return { pred: cuoi === 'T' ? 'X' : 'T', score: 85, reason: `Bẻ nhẹ cầu 1-1 tại tay 5 (${mau})` };
            if (current_len == 6) return { pred: cuoi === 'T' ? 'X' : 'T', score: 90, reason: `Ôm thêm tay 6 rồi bẻ cầu 1-1 (${mau})` };
            return { pred: cuoi, score: 72, reason: 'Không rõ mẫu → Theo tay gần nhất' };
          }
        }
      }
    }
    for (let loai in cau_mau) {
      const arr = cau_mau[loai];
      if (arr.some(a => pattern.endsWith(a))) {
        return { pred: cuoi === 'T' ? 'X' : 'T', score: 90, reason: `Phát hiện cầu ${loai}` };
      }
    }
    if (data_kq.length >= 6) {
      const last6 = data_kq.slice(-6);
      for (let i = 2; i < 6; i++) {
        if (i * 2 <= last6.length) {
          const seq = last6.slice(-i * 2);
          const alt1 = [], alt2 = [];
          for (let j = 0; j < i * 2; j++) { alt1.push(j % 2 === 0 ? 'T' : 'X'); alt2.push(j % 2 === 0 ? 'X' : 'T'); }
          if (seq.join('') === alt1.join('') || seq.join('') === alt2.join('')) {
            return { pred: (cuoi === 'X') ? 'T' : 'X', score: 90, reason: `Bẻ cầu 1-1 (${i * 2} tay)` };
          }
        }
      }
    }
    if (dem_sai >= 3) return { pred: cuoi === 'T' ? 'X' : 'T', score: 88, reason: 'Sai 3 lần → Đổi chiều' };
    if (data_kq.length >= 3 && pattern_sai.hasOwnProperty(data_kq.slice(-3).join(','))) return { pred: cuoi === 'T' ? 'X' : 'T', score: 86, reason: 'Mẫu sai cũ' };
    if (chenh >= 3) {
      const uu = countsObj.T > countsObj.X ? 'T' : 'X';
      return { pred: uu, score: 84, reason: `Lệch ${chenh} cầu → Ưu tiên ${uu}` };
    }
    return { pred: cuoi, score: 72, reason: 'Không rõ mẫu → Theo tay gần nhất' };
  } catch (e) {
    return { pred: 'T', score: 50, reason: 'Lỗi trong du_doan_js: ' + (e.message || e) };
  }
}

// Predictor Service
class PredictorService {
  constructor(history) {
    this.history = history || [];
    this.ensemble = new Ensemble();
    this.ensemble.trainAll(seqFromHistory(this.history));
    this.predHistory = [];
    this.data_store = {};
    this.dem_sai = 0;
    this.pattern_sai = {};
    this.diem_lich_su = [];
  }
  predict() {
    const seq = seqFromHistory(this.history);
    const totals = this.history.map(h => h.Tong).filter(x => x !== null);
    const roadType = classifyRoad(seq);
    const modelOut = this.ensemble.predictProba(seq);
    const top = Math.max(modelOut.distribution.T, modelOut.distribution.X);
    const entropy = - (modelOut.distribution.T * Math.log2(modelOut.distribution.T + 1e-9) + modelOut.distribution.X * Math.log2(modelOut.distribution.X + 1e-9));
    const weightEntropy = -Object.values(this.ensemble.weights).reduce((s, w) => s + w * Math.log2(w + 1e-9), 0);
    const weightConcentration = 1 - (weightEntropy / Math.log2(CONFIG.MODELS.length));
    const conf = clamp(CONFIG.BASE_CONFIDENCE * 0.3 + top * 0.6 + weightConcentration * 0.1 - (entropy * 0.05), 0, 1);
    const predicted = modelOut.distribution.T >= modelOut.distribution.X ? 'T' : 'X';
    const reasonPieces = [];
    const modelScores = {};
    CONFIG.MODELS.forEach(m => {
      const p = modelOut.modelProbas[m][predicted];
      modelScores[m] = (this.ensemble.weights[m] || 0) * p;
    });
    const topModel = Object.keys(modelScores).reduce((a, b) => modelScores[a] > modelScores[b] ? a : b);
    reasonPieces.push(`Top model: ${topModel} (w=${(this.ensemble.weights[topModel] || 0).toFixed(3)})`);
    reasonPieces.push(`Road type: ${roadType}`);
    const runInfo = computeRunLength(seq);
    reasonPieces.push(`Run: ${runInfo.run} of ${runInfo.value || '-'}`);
    const pat = this.ensemble.models.pattern.detectPattern(seq);
    if (pat.type !== 'none') reasonPieces.push(`Pattern detected: ${pat.type} (str=${pat.strength.toFixed(2)})`);
    if (runInfo.run >= CONFIG.RUN_WINDOW_SHORT) reasonPieces.push('Long run → tăng khả năng bẻ');
    else reasonPieces.push('Short run/mixed → momentum ủng hộ tiếp tục');
    const manual = matchManualPattern(totals);
    let manualObj = null;
    if (manual) {
      manualObj = { pred: manual.pred, note: manual.note, weight: 0.9 };
      reasonPieces.push(`Manual pattern matched: ${manual.note}`);
    }
    const last = this.history.length ? this.history[this.history.length - 1] : null;
    const xx_str = last && last.Xuc_xac_1 ? `${last.Xuc_xac_1}-${last.Xuc_xac_2}-${last.Xuc_xac_3}` : '';
    const human_seq_labels = this.history.map(h => h.Ket_qua ? (h.Ket_qua === 'Tài' ? 'T' : 'X') : null).filter(x => x);
    const duObj = du_doan_js(human_seq_labels, this.dem_sai, this.pattern_sai, xx_str, this.diem_lich_su, this.data_store);
    const ensembleProb = modelOut.distribution;
    const ensemblePred = ensembleProb.T >= ensembleProb.X ? 'T' : 'X';
    let weights = { ensemble: 0.45, du: 0.35, manual: 0.20 };
    if (manualObj) {
      weights.manual = 0.4; weights.ensemble = 0.35; weights.du = 0.25;
    }
    const scoreT = weights.ensemble * ensembleProb.T + weights.du * (duObj.pred === 'T' ? duObj.score / 100 : (100 - duObj.score) / 100) + (manualObj ? (weights.manual * (manualObj.pred === 'T' ? manualObj.weight : (1 - manualObj.weight))) : 0);
    const scoreX = weights.ensemble * ensembleProb.X + weights.du * (duObj.pred === 'X' ? duObj.score / 100 : (100 - duObj.score) / 100) + (manualObj ? (weights.manual * (manualObj.pred === 'X' ? manualObj.weight : (1 - manualObj.weight))) : 0);
    const norm = scoreT + scoreX || 1;
    const finalT = scoreT / norm;
    const finalX = scoreX / norm;
    const finalPred = finalT >= finalX ? 'T' : 'X';
    const finalConf = clamp(Math.max(finalT, finalX), 0, 1);
    const reason = [
      `Ensemble: ${ensemblePred} (pT=${ensembleProb.T.toFixed(3)}, pX=${ensembleProb.X.toFixed(3)})`,
      `du_doan: ${duObj.pred} (score=${duObj.score}) - ${duObj.reason}`,
      manualObj ? `Manual: ${manualObj.pred} (${manualObj.note})` : null,
      `Fusion weights: ensemble=${weights.ensemble}, du=${weights.du}, manual=${weights.manual || 0}`,
      `Final fusion: pT=${finalT.toFixed(3)}, pX=${finalX.toFixed(3)}`
    ].filter(x => x).join(' | ');
    return {
      timestamp: nowStr(),
      prediction: finalPred === 'T' ? 'Tài' : 'Xỉu',
      confidence: Math.round(finalConf * 10000) / 100,
      distribution: { T: finalT, X: finalX },
      ensemble: modelOut,
      du_doan: duObj,
      manual: manualObj,
      reason,
      roadType,
      runInfo,
      history_len: this.history.length,
      last_round: last
    };
  }
  learn(actualRound) {
    this.history.push(actualRound);
    if (this.history.length > CONFIG.MAX_HISTORY_STORE) this.history = this.history.slice(-CONFIG.MAX_HISTORY_STORE);
    this.ensemble.trainAll(seqFromHistory(this.history));
    const seqBefore = seqFromHistory(this.history.slice(0, -1));
    const actual = actualRound.Ket_qua ? (actualRound.Ket_qua === 'Tài' ? 'T' : 'X') : 'T';
    this.ensemble.updateWeights(seqBefore, actual);
  }
}

function classifyRoad(seq) {
  const cShort = counts(last(seq, 12));
  const rateT = cShort.T / (cShort.T + cShort.X || 1);
  const r = computeRunLength(seq);
  const tail = last(seq, 6).join('');
  if (/^([TX])([TX])\1\2\1\2$/.test(tail)) return 'zigzag';
  if (r.run >= 6) return 'streaky';
  if (Math.abs(rateT - 0.5) < 0.08) return 'flat';
  if (rateT > 0.6) return 'trending_T';
  if (rateT < 0.4) return 'trending_X';
  return 'mixed';
}

async function fetchApiOnce() {
  try {
    const resp = await axios.get(CONFIG.API_URL);
    if (!resp.status === 200) throw new Error('API lỗi ' + resp.status);
    const data = resp.data;
    let arr = [];
    if (Array.isArray(data)) arr = data;
    else if (data && data.data && Array.isArray(data.data)) arr = data.data;
    else if (data && data.result && Array.isArray(data.result)) arr = data.result;
    else if (data) arr = [data];
    const mapped = arr.map(r => {
      const X1 = r.Xuc_xac_1 !== undefined ? r.Xuc_xac_1 : (r.x1 !== undefined ? r.x1 : (r.d1 !== undefined ? r.d1 : null));
      const X2 = r.Xuc_xac_2 !== undefined ? r.Xuc_xac_2 : (r.x2 !== undefined ? r.x2 : (r.d2 !== undefined ? r.d2 : null));
      const X3 = r.Xuc_xac_3 !== undefined ? r.Xuc_xac_3 : (r.x3 !== undefined ? r.x3 : (r.d3 !== undefined ? r.d3 : null));
      const total = r.Tong !== undefined ? r.Tong : (r.total !== undefined ? r.total : (X1 !== null && X2 !== null && X3 !== null ? (X1 + X2 + X3) : null));
      const ket = r.Ket_qua !== undefined ? r.Ket_qua : (r.result !== undefined ? (r.result === 'T' ? 'Tài' : 'Xỉu') : null);
      return {
        Phien: r.Phien || r.id || r.Phien_hien_tai || null,
        Xuc_xac_1: X1, Xuc_xac_2: X2, Xuc_xac_3: X3,
        Tong: total, Ket_qua: ket,
        raw: r
      };
    });
    return mapped;
  } catch (e) {
    throw e;
  }
}

// API Endpoint
app.get('/api/taixiu/sunwin', async (req, res) => {
  try {
    // Fetch data from the original API
    const mapped = await fetchApiOnce();
    if (!mapped || !mapped.length) {
      return res.status(500).json({ error: 'API trả về rỗng' });
    }

    // Update history
    const existingSet = new Set(history.map(h => h.Phien));
    for (let r of mapped) {
      if (r.Phien && !existingSet.has(r.Phien)) {
        history.push(r);
      }
    }
    if (history.length > CONFIG.MAX_HISTORY_STORE) {
      history = history.slice(-CONFIG.MAX_HISTORY_STORE);
    }

    // Initialize PredictorService
    const service = new PredictorService(history);
    const prediction = service.predict();

    // Get the latest round
    const lastRound = history[history.length - 1];

    // Map to required response format
    const response = {
      Phien: lastRound.Phien,
      Phien_sau: lastRound.Phien + 1,
      d1: lastRound.Xuc_xac_1,
      d2: lastRound.Xuc_xac_2,
      d3: lastRound.Xuc_xac_3,
      Tong: lastRound.Tong,
      Result: lastRound.Ket_qua,
      Du_doan: prediction.prediction,
      Do_tin_cay: prediction.confidence,
      Giai_thich: prediction.reason,
      Pattern: lastRound.raw?.Pattern || seqFromHistory(history.slice(-20)).join('').toLowerCase().replace(/t/g, 'tài').replace(/x/g, 'xỉu'),
      id: '@ANALYSIS TỚI CHƠI 🤟'
    };

    res.json(response);
  } catch (e) {
    res.status(500).json({ error: 'Lỗi khi lấy dữ liệu: ' + (e.message || e) });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
