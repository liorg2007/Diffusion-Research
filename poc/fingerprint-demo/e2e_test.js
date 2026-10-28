'use strict';
// End-to-end check against a running server (node server.js): one multipart upload per laundering condition, then every sample at every
// condition (JSON sample_id), printing hit rates. Node 12, built-ins only.   node e2e_test.js [port]
const http = require('http');
const fs = require('fs');
const path = require('path');
const PORT = parseInt(process.argv[2] || '5173', 10);

function req(method, p, body, headers) {
  return new Promise(function (resolve, reject) {
    const r = http.request({ host: '127.0.0.1', port: PORT, method: method, path: p, headers: headers || {} }, function (res) {
      const ch = []; res.on('data', function (c) { ch.push(c); });
      res.on('end', function () { let j = null; try { j = JSON.parse(Buffer.concat(ch).toString()); } catch (e) { j = null; } resolve({ status: res.statusCode, json: j }); });
    });
    r.on('error', reject); if (body) r.write(body); r.end();
  });
}
function multipart(file, cond) {
  const b = 'XBOUNDARY' + Date.now();
  const head = Buffer.from('--' + b + '\r\nContent-Disposition: form-data; name="condition"\r\n\r\n' + cond + '\r\n--' + b + '\r\nContent-Disposition: form-data; name="file"; filename="' + path.basename(file) + '"\r\nContent-Type: image/jpeg\r\n\r\n');
  const data = fs.readFileSync(file);
  return { body: Buffer.concat([head, data, Buffer.from('\r\n--' + b + '--\r\n')]), type: 'multipart/form-data; boundary=' + b };
}
async function main() {
  const sj = (await req('GET', '/api/samples')).json.samples;
  const conds = ['clean', 'jpeg75_resize05', 'social_sim', 'screenshot_sim'];
  console.log('--- multipart uploads');
  const f = path.join(__dirname, 'samples', sj[0].file);
  for (const c of conds.slice(0, 3)) {
    const m = multipart(f, c);
    const r = await req('POST', '/api/analyze', m.body, { 'Content-Type': m.type, 'Content-Length': m.body.length });
    const o = r.json;
    console.log(c, r.status, o.ok, 'top=' + o.top, o.top_prob.toFixed(3), 'uncertain=' + o.uncertain, 'ai=' + o.ai.prob.toFixed(3), o.ai.verdict, o.timing.total_ms + 'ms', 'size', o.laundered_size.join('x'));
  }
  const bad = await req('POST', '/api/analyze', Buffer.from('hello world not an image'), { 'Content-Type': 'application/json' });
  console.log('bad body ->', bad.status);
  const tmpLeft = fs.readdirSync(path.join(__dirname, 'tmp')).filter(function (x) { return /^up_/.test(x); });
  console.log('temp files left:', tmpLeft.length);
  console.log('--- sample hit rates (n fakes / reals):');
  const meta = (await req('GET', '/api/meta')).json;
  for (const c of conds) {
    let gOk = 0, nf = 0, dOk = 0, rOk = 0, nr = 0, unc = 0;
    for (const s of sj) {
      const o = (await req('POST', '/api/analyze', JSON.stringify({ sample_id: s.id, condition: c }), { 'Content-Type': 'application/json' })).json;
      if (s.kind === 'fake') { nf++; gOk += o.truth.generator_correct ? 1 : 0; dOk += o.truth.detector_correct ? 1 : 0; unc += o.uncertain ? 1 : 0; } else { nr++; rOk += o.truth.detector_correct ? 1 : 0; }
    }
    console.log(c.padEnd(16), 'generator named right ' + gOk + '/' + nf + ' (' + (gOk / nf).toFixed(2) + ') vs held-out bacc ' + meta.attribution.heldout_bacc_fold[c].toFixed(3), '| fakes flagged ' + dOk + '/' + nf, '| reals unflagged ' + rOk + '/' + nr, '| uncertain ' + unc + '/' + nf);
  }
}
main().catch(function (e) { console.error(e); process.exit(1); });
