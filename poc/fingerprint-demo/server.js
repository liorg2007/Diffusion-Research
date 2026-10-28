'use strict';
// Fingerprint demo server: Node built-ins only (Node 12 compatible), 127.0.0.1:5173.
//   GET  /                  public/ (static)
//   GET  /api/status        worker state
//   GET  /api/samples       samples/samples.json
//   GET  /samples/<file>    sample image (whitelisted by samples.json)
//   GET  /api/findings      findings.json (build_findings.py)
//   GET  /api/meta          models/meta.json (provenance + held-out metrics)
//   POST /api/analyze       multipart (file + condition) | JSON {image_base64, condition} | JSON {sample_id, condition}
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const crypto = require('crypto');
const childProcess = require('child_process');

const HERE = __dirname;
const ROOT = path.resolve(HERE, '..', '..');
const PUBLIC = path.join(HERE, 'public');
const TMP = path.join(HERE, 'tmp');
const SAMPLES = path.join(HERE, 'samples');
const MODELS = path.join(HERE, 'models');
const PYTHON = process.env.DEMO_PYTHON || path.join(ROOT, 'env', '.venv', 'bin', 'python');
const PORT = parseInt(process.env.PORT || '5173', 10);
const HOST = '127.0.0.1';
const MAX_UPLOAD = 15 * 1024 * 1024;
const MAX_BODY = MAX_UPLOAD + 1024 * 1024; // multipart / base64 overhead is checked again on the decoded payload
const REQUEST_TIMEOUT_MS = 180000;
const CONDITIONS = ['clean', 'jpeg75_resize05', 'social_sim', 'screenshot_sim'];
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.png': 'image/png', '.ico': 'image/x-icon'
};

fs.mkdirSync(TMP, { recursive: true });

// ---------------------------------------------------------------- worker supervision + queue
const worker = { proc: null, ready: false, device: null, loadS: null, restarts: 0, buf: '', starting: false };
const queue = []; // {id, payload, resolve, reject, tmpFile}
let inflight = null;
let inflightTimer = null;
let readyWaiters = [];

function log() {
  const a = Array.prototype.slice.call(arguments);
  a.unshift(new Date().toISOString());
  console.log.apply(console, a);
}

function startWorker() {
  if (worker.proc || worker.starting) return;
  worker.starting = true;
  worker.ready = false;
  worker.buf = '';
  const args = ['-I', path.join(HERE, 'worker.py')];
  log('starting worker:', PYTHON, args.join(' '));
  const p = childProcess.spawn(PYTHON, args, { cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'] });
  worker.proc = p;
  worker.starting = false;
  p.stdout.setEncoding('utf8');
  p.stdout.on('data', function (chunk) {
    worker.buf += chunk;
    let i;
    while ((i = worker.buf.indexOf('\n')) >= 0) {
      const line = worker.buf.slice(0, i).trim();
      worker.buf = worker.buf.slice(i + 1);
      if (line) onWorkerLine(line);
    }
  });
  p.stderr.setEncoding('utf8');
  p.stderr.on('data', function (d) { process.stderr.write('[worker] ' + d); });
  p.stdin.on('error', function () { /* worker died: handled in exit */ });
  p.on('exit', function (code, sig) {
    log('worker exited', code, sig);
    worker.proc = null;
    worker.ready = false;
    if (inflightTimer) { clearTimeout(inflightTimer); inflightTimer = null; }
    if (inflight) { finish(inflight, new Error('worker exited during the request'), null); inflight = null; }
    if (!shuttingDown) {
      worker.restarts += 1;
      const delay = Math.min(10000, 1000 * worker.restarts);
      setTimeout(startWorker, delay);
    }
  });
}

function onWorkerLine(line) {
  let msg;
  try { msg = JSON.parse(line); } catch (e) { log('bad worker line:', line.slice(0, 200)); return; }
  if (msg.event === 'ready') {
    worker.ready = true;
    worker.restarts = 0;
    worker.device = msg.device;
    worker.loadS = msg.load_s;
    log('worker ready on', msg.device);
    const w = readyWaiters; readyWaiters = [];
    w.forEach(function (f) { f(); });
    pump();
    return;
  }
  if (inflight && msg.id === inflight.id) {
    if (inflightTimer) { clearTimeout(inflightTimer); inflightTimer = null; }
    const job = inflight;
    inflight = null;
    finish(job, null, msg);
    pump();
  }
}

function finish(job, err, msg) {
  if (job.tmpFile) fs.unlink(job.tmpFile, function () {});
  if (err) job.reject(err); else job.resolve(msg);
}

function pump() {
  if (inflight || !worker.ready || !worker.proc || queue.length === 0) return;
  inflight = queue.shift();
  const job = inflight;
  const req = Object.assign({ id: job.id }, job.payload);
  try {
    worker.proc.stdin.write(JSON.stringify(req) + '\n');
  } catch (e) {
    inflight = null;
    finish(job, e, null);
    return;
  }
  inflightTimer = setTimeout(function () {
    log('request timed out, restarting worker');
    if (worker.proc) worker.proc.kill('SIGKILL');
  }, REQUEST_TIMEOUT_MS);
}

function submit(payload, tmpFile) {
  return new Promise(function (resolve, reject) {
    if (queue.length >= 20) { if (tmpFile) fs.unlink(tmpFile, function () {}); reject(new Error('queue full')); return; }
    const id = crypto.randomBytes(6).toString('hex');
    queue.push({ id: id, payload: payload, resolve: resolve, reject: reject, tmpFile: tmpFile });
    pump();
  });
}

function waitReady(ms) {
  return new Promise(function (resolve, reject) {
    if (worker.ready) return resolve();
    const t = setTimeout(function () { reject(new Error('worker is not ready yet (model still loading)')); }, ms);
    readyWaiters.push(function () { clearTimeout(t); resolve(); });
  });
}

// ---------------------------------------------------------------- helpers
function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

function readJsonFile(p, cb) {
  fs.readFile(p, 'utf8', function (err, s) {
    if (err) return cb(err);
    try { cb(null, JSON.parse(s)); } catch (e) { cb(e); }
  });
}

function readBody(req, limit, cb) {
  const chunks = [];
  let size = 0;
  let done = false;
  req.on('data', function (c) {
    if (done) return;
    size += c.length;
    if (size > limit) { done = true; cb(Object.assign(new Error('payload too large (max 15 MB)'), { status: 413 })); req.resume(); return; }
    chunks.push(c);
  });
  req.on('end', function () { if (!done) { done = true; cb(null, Buffer.concat(chunks)); } });
  req.on('error', function (e) { if (!done) { done = true; cb(e); } });
}

// minimal multipart/form-data parser -> {fields:{}, file:{name, data}}
function parseMultipart(buf, boundary) {
  const delim = Buffer.from('--' + boundary);
  const out = { fields: {}, file: null };
  let pos = buf.indexOf(delim);
  while (pos >= 0) {
    const next = buf.indexOf(delim, pos + delim.length);
    if (next < 0) break;
    let part = buf.slice(pos + delim.length, next);
    if (part.slice(0, 2).toString() === '\r\n') part = part.slice(2);
    if (part.slice(-2).toString() === '\r\n') part = part.slice(0, -2);
    const sep = part.indexOf('\r\n\r\n');
    if (sep > 0) {
      const head = part.slice(0, sep).toString('utf8');
      const data = part.slice(sep + 4);
      const nm = /name="([^"]*)"/.exec(head);
      const fn = /filename="([^"]*)"/.exec(head);
      if (fn) out.file = { name: fn[1], data: data };
      else if (nm) out.fields[nm[1]] = data.toString('utf8');
    }
    pos = next;
  }
  return out;
}

function looksLikeImage(b) {
  if (b.length < 12) return false;
  if (b[0] === 0xff && b[1] === 0xd8) return true; // JPEG
  if (b.slice(0, 8).toString('hex') === '89504e470d0a1a0a') return true; // PNG
  const h = b.slice(0, 12).toString('latin1');
  if (h.slice(0, 4) === 'RIFF' && h.slice(8, 12) === 'WEBP') return true;
  if (h.slice(0, 3) === 'GIF' || h.slice(0, 2) === 'BM') return true;
  if (h.slice(0, 2) === 'II' || h.slice(0, 2) === 'MM') return true; // TIFF
  return false;
}

function loadSamples(cb) {
  readJsonFile(path.join(SAMPLES, 'samples.json'), cb);
}

function truthCheck(sample, out) {
  if (!sample || !out || !out.ok) return null;
  const r = { kind: sample.kind, true_label: sample.true_label, true: sample.true };
  if (sample.kind === 'fake') {
    r.detector_correct = out.ai.verdict === 'ai';
    r.generator_correct = out.top === sample.true;
  } else {
    r.detector_correct = out.ai.verdict !== 'ai';
    r.generator_correct = null;
  }
  return r;
}

function handleAnalyze(req, res) {
  const ctype = req.headers['content-type'] || '';
  readBody(req, MAX_BODY, function (err, body) {
    if (err) return sendJson(res, err.status || 400, { ok: false, error: err.message });
    let condition = 'jpeg75_resize05';
    let sampleId = null;
    let data = null;
    let jsonBody = null;
    try {
      if (/^multipart\/form-data/i.test(ctype)) {
        const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(ctype);
        if (!m) throw new Error('multipart boundary missing');
        const mp = parseMultipart(body, m[1] || m[2]);
        if (mp.fields.condition) condition = mp.fields.condition;
        if (mp.fields.sample_id) sampleId = mp.fields.sample_id;
        data = mp.file ? mp.file.data : null;
      } else {
        jsonBody = JSON.parse(body.toString('utf8') || '{}');
        if (jsonBody.condition) condition = jsonBody.condition;
        if (jsonBody.sample_id) sampleId = String(jsonBody.sample_id);
        if (jsonBody.image_base64) data = Buffer.from(String(jsonBody.image_base64).replace(/^data:[^,]*,/, ''), 'base64');
      }
    } catch (e) {
      return sendJson(res, 400, { ok: false, error: 'bad request: ' + e.message });
    }
    if (CONDITIONS.indexOf(condition) < 0) return sendJson(res, 400, { ok: false, error: 'unknown condition' });
    if (data && data.length > MAX_UPLOAD) return sendJson(res, 413, { ok: false, error: 'image larger than 15 MB' });

    const go = function (sample, payload, tmpFile) {
      waitReady(120000).then(function () { return submit(payload, tmpFile); }).then(function (out) {
        if (out && out.ok) { out.sample = sample ? { id: sample.id, kind: sample.kind, true: sample.true, true_label: sample.true_label, dataset: sample.dataset, license: sample.license } : null; out.truth = truthCheck(sample, out); }
        sendJson(res, out && out.ok ? 200 : 422, out);
      }).catch(function (e) {
        if (tmpFile) fs.unlink(tmpFile, function () {});
        sendJson(res, 503, { ok: false, error: e.message });
      });
    };

    if (sampleId && !data) {
      return loadSamples(function (e, sj) {
        const s = !e && sj.samples.filter(function (x) { return x.id === sampleId; })[0];
        if (!s) return sendJson(res, 404, { ok: false, error: 'unknown sample' });
        go(s, { image_path: path.join(SAMPLES, s.file), condition: condition }, null);
      });
    }
    if (!data || data.length === 0) return sendJson(res, 400, { ok: false, error: 'no image in the request' });
    if (!looksLikeImage(data)) return sendJson(res, 415, { ok: false, error: 'not a recognised image file (JPEG, PNG, WebP, GIF, BMP, TIFF)' });
    const tmpFile = path.join(TMP, 'up_' + crypto.randomBytes(8).toString('hex') + '.bin');
    fs.writeFile(tmpFile, data, function (we) {
      if (we) return sendJson(res, 500, { ok: false, error: 'cannot write temp file' });
      if (sampleId) {
        loadSamples(function (e, sj) {
          const s = !e && sj.samples.filter(function (x) { return x.id === sampleId; })[0];
          go(s || null, { image_path: tmpFile, condition: condition }, tmpFile);
        });
      } else {
        go(null, { image_path: tmpFile, condition: condition }, tmpFile);
      }
    });
  });
}

function serveStatic(root, rel, res, whitelist) {
  let p;
  try { p = decodeURIComponent(rel); } catch (e) { res.writeHead(400); return res.end('bad path'); }
  if (p.indexOf('\0') >= 0) { res.writeHead(400); return res.end('bad path'); }
  if (p === '' || p.slice(-1) === '/') p += 'index.html';
  const full = path.normalize(path.join(root, p));
  if (full.indexOf(root + path.sep) !== 0) { res.writeHead(403); return res.end('forbidden'); }
  if (whitelist && whitelist.indexOf(path.basename(full)) < 0) { res.writeHead(404); return res.end('not found'); }
  fs.stat(full, function (err, st) {
    if (err || !st.isFile()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(full).toLowerCase()] || 'application/octet-stream', 'Content-Length': st.size, 'Cache-Control': 'no-cache' });
    fs.createReadStream(full).pipe(res);
  });
}

const server = http.createServer(function (req, res) {
  const u = url.parse(req.url);
  const pathname = u.pathname || '/';
  if (req.method === 'POST' && pathname === '/api/analyze') return handleAnalyze(req, res);
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405); return res.end('method not allowed'); }
  if (pathname === '/api/status') {
    return sendJson(res, 200, { ready: worker.ready, device: worker.device, load_s: worker.loadS, queue: queue.length + (inflight ? 1 : 0), restarts: worker.restarts });
  }
  if (pathname === '/api/samples') {
    return loadSamples(function (e, sj) { if (e) return sendJson(res, 404, { error: 'samples not built: run make_samples.py', samples: [] }); sendJson(res, 200, sj); });
  }
  if (pathname === '/api/findings') {
    return readJsonFile(path.join(HERE, 'findings.json'), function (e, j) { if (e) return sendJson(res, 404, { error: 'findings.json missing: run build_findings.py' }); sendJson(res, 200, j); });
  }
  if (pathname === '/api/meta') {
    return readJsonFile(path.join(MODELS, 'meta.json'), function (e, j) { if (e) return sendJson(res, 404, { error: 'models missing: run train_models.py' }); sendJson(res, 200, j); });
  }
  if (pathname.indexOf('/samples/') === 0) {
    return loadSamples(function (e, sj) {
      if (e) { res.writeHead(404); return res.end('not found'); }
      serveStatic(SAMPLES, pathname.slice('/samples/'.length), res, sj.samples.map(function (s) { return s.file; }));
    });
  }
  serveStatic(PUBLIC, pathname.slice(1), res, null);
});

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  log('shutting down');
  if (worker.proc) { try { worker.proc.stdin.end(); worker.proc.kill('SIGTERM'); } catch (e) { /* ignore */ } }
  fs.readdir(TMP, function (e, files) { (files || []).forEach(function (f) { if (/^up_/.test(f)) fs.unlink(path.join(TMP, f), function () {}); }); });
  server.close();
  setTimeout(function () { process.exit(0); }, 500);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// clean stale temp uploads from a previous crash
fs.readdir(TMP, function (e, files) { (files || []).forEach(function (f) { if (/^up_/.test(f)) fs.unlink(path.join(TMP, f), function () {}); }); });
server.listen(PORT, HOST, function () { log('listening on http://' + HOST + ':' + PORT); startWorker(); });
