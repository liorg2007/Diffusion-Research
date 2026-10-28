/* Fingerprint Lab front end: vanilla JS, no dependencies. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var esc = Charts.esc, f3 = Charts.f3, pct = Charts.pct;
  var GEN = { 'dalle2': 'DALL-E 2', 'dalle3': 'DALL-E 3', 'firefly': 'Adobe Firefly', 'midjourney-v5': 'Midjourney v5', 'stable-diffusion-2': 'Stable Diffusion 2', 'stable-diffusion-xl': 'Stable Diffusion XL', 'sd3_large': 'Stable Diffusion 3', 'flux': 'FLUX' };
  var CONDS = [
    { id: 'clean', name: 'Clean', info: 'No extra processing (the image is still scaled so its short side is 512 px and centre-cropped, as in every experiment).' },
    { id: 'jpeg75_resize05', name: 'JPEG-75 + half size', info: 'The project\'s primary test: shrink to 50%, then JPEG quality 75. Erases most low-level fingerprints.' },
    { id: 'social_sim', name: 'Social-media sim', info: 'Random resize to 70-100% then JPEG quality 70-100, like a platform re-upload.' },
    { id: 'screenshot_sim', name: 'Screenshot sim (never trained on)', info: 'Rescaled by 0.5, 0.67 or 1.25 (bilinear) and a thin border cropped, no JPEG. A laundering the probes never saw in training.' }
  ];
  var state = { cond: 'jpeg75_resize05', cur: null, meta: null, find: null, samples: [], busy: false, tally: {}, seq: 0 };

  /* ------------------------------------------------------------ theme + tabs */
  function theme() { try { return localStorage.getItem('theme'); } catch (e) { return null; } }
  function setTheme(t) { if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme'); try { if (t) localStorage.setItem('theme', t); } catch (e) { /* ignore */ } renderHeat(); }
  if (theme()) document.documentElement.setAttribute('data-theme', theme());
  $('themeBtn').onclick = function () {
    var dark = document.documentElement.getAttribute('data-theme') === 'dark' || (!document.documentElement.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    setTheme(dark ? 'light' : 'dark');
  };
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addListener(function () { renderHeat(); });
  var tabs = ['analyze', 'findings', 'phone', 'caveats'];
  function showTab(n) {
    tabs.forEach(function (t) { var on = t === n; $('tab-' + t).setAttribute('aria-selected', on); $('p-' + t).classList.toggle('active', on); });
    try { history.replaceState(null, '', '#' + n); } catch (e) { /* ignore */ }
    if (n === 'phone') renderHeat();
  }
  tabs.forEach(function (t) { $('tab-' + t).onclick = function () { showTab(t); }; });

  /* ------------------------------------------------------------ boot */
  function getJson(u) { return fetch(u).then(function (r) { return r.json(); }); }
  function boot() {
    renderConds();
    Promise.all([getJson('/api/meta').catch(function () { return null; }), getJson('/api/findings').catch(function () { return null; }), getJson('/api/samples').catch(function () { return { samples: [] }; })]).then(function (r) {
      state.meta = r[0] && !r[0].error ? r[0] : null;
      state.find = r[1] && !r[1].error ? r[1] : null;
      state.samples = (r[2] && r[2].samples) || [];
      renderSamples();
      if (state.find) { renderFindings(); setupPhone(); }
      renderCaveats();
      var h = (location.hash || '').slice(1);
      if (tabs.indexOf(h) >= 0) showTab(h);
      var q = {}; (location.search || '').slice(1).split('&').forEach(function (kv) { var p = kv.split('='); if (p[0]) q[p[0]] = decodeURIComponent(p[1] || ''); });
      if (q.cond && CONDS.some(function (c) { return c.id === q.cond; })) { state.cond = q.cond; renderConds(); }
      if (q.sample && state.samples.some(function (s) { return s.id === q.sample; })) pickSample(q.sample);
    });
    pollStatus();
  }
  function pollStatus() {
    getJson('/api/status').then(function (s) {
      var el = $('status');
      el.className = 'pill ' + (s.ready ? 'ok' : '');
      $('statusText').textContent = s.ready ? 'model ready · ' + s.device : 'loading model…';
      setTimeout(pollStatus, s.ready ? 8000 : 1500);
    }).catch(function () { $('status').className = 'pill bad'; $('statusText').textContent = 'server offline'; setTimeout(pollStatus, 3000); });
  }

  /* ------------------------------------------------------------ analyze: inputs */
  function renderConds() {
    $('conds').innerHTML = CONDS.map(function (c) {
      return '<label class="opt' + (c.id === state.cond ? ' sel' : '') + '"><input type="radio" name="cond" value="' + c.id + '"' + (c.id === state.cond ? ' checked' : '') + '><b>' + esc(c.name) + '</b><span>' + esc(c.info) + '</span></label>';
    }).join('');
    Array.prototype.forEach.call(document.querySelectorAll('input[name=cond]'), function (i) {
      i.onchange = function () {
        state.cond = i.value;
        Array.prototype.forEach.call(document.querySelectorAll('.opt'), function (o) { o.classList.toggle('sel', o.querySelector('input').checked); });
        if (state.cur) analyze();
      };
    });
  }
  function renderSamples() {
    if (!state.samples.length) { $('samples').innerHTML = '<p class="small muted" style="margin-top:10px">No samples found. Run <code>python make_samples.py</code>.</p>'; return; }
    function hsh(str) { var h = 7; for (var i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) % 1000003; return h; }
    var byHash = function (a, b) { return hsh(a.id) - hsh(b.id); };
    var fakes = state.samples.filter(function (s) { return s.kind === 'fake'; }).sort(byHash), reals = state.samples.filter(function (s) { return s.kind === 'real'; }).sort(byHash);
    function grid(list) {
      return '<div class="sgrid">' + list.map(function (s) {
        return '<button data-id="' + s.id + '" aria-pressed="false" title="Sample (label hidden until analysed)"><img loading="lazy" src="/samples/' + s.file + '" alt="Sample image"></button>';
      }).join('') + '</div>';
    }
    $('samples').innerHTML = '<div class="sgroup"><div class="label">Samples: AI-generated (hidden label)</div>' + grid(fakes) + '</div><div class="sgroup"><div class="label">Samples: real photos</div>' + grid(reals) + '</div><p class="tiny faint" style="margin-top:8px">Samples come from images that neither model trained on (shuffled; the true label is revealed in the result).</p>';
    Array.prototype.forEach.call($('samples').querySelectorAll('button'), function (b) {
      b.onclick = function () { pickSample(b.getAttribute('data-id')); };
    });
  }
  function markPressed(id) { Array.prototype.forEach.call(document.querySelectorAll('.sgrid button'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-id') === id); }); }
  function pickSample(id) {
    var s = state.samples.filter(function (x) { return x.id === id; })[0];
    state.cur = { type: 'sample', sample: s, url: '/samples/' + s.file };
    markPressed(id); analyze();
  }
  var drop = $('drop');
  ['dragenter', 'dragover'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (ev) { var f = ev.dataTransfer.files[0]; if (f) pickFile(f); });
  $('file').onchange = function () { if (this.files[0]) pickFile(this.files[0]); this.value = ''; };
  function pickFile(f) {
    if (f.size > 15 * 1024 * 1024) { alert('That file is larger than 15 MB.'); return; }
    if (state.cur && state.cur.type === 'file') URL.revokeObjectURL(state.cur.url);
    state.cur = { type: 'file', file: f, url: URL.createObjectURL(f) };
    markPressed(null); analyze();
  }

  /* ------------------------------------------------------------ analyze: run + render */
  function post(cur, cond) {
    if (cur.type === 'sample') return fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sample_id: cur.sample.id, condition: cond }) }).then(function (r) { return r.json(); });
    var fd = new FormData(); fd.append('condition', cond); fd.append('file', cur.file, cur.file.name || 'upload');
    return fetch('/api/analyze', { method: 'POST', body: fd }).then(function (r) { return r.json(); });
  }
  function analyze() {
    var cur = state.cur, my = ++state.seq;
    $('emptyCard').hidden = true; $('result').hidden = false; $('result').style.opacity = .55;
    $('aiChip').innerHTML = '<span class="spin"></span>'; $('genChip').innerHTML = '<span class="spin"></span>';
    $('imgOrig').src = cur.url;
    post(cur, state.cond).then(function (out) {
      if (my !== state.seq) return;
      $('result').style.opacity = 1;
      if (!out.ok) { $('aiChip').innerHTML = '<span class="chip bad">Error</span>'; $('aiCaveat').textContent = out.error || 'request failed'; $('genChip').innerHTML = ''; return; }
      render(out, cur);
    }).catch(function (e) { $('result').style.opacity = 1; $('aiChip').innerHTML = '<span class="chip bad">Error</span>'; $('aiCaveat').textContent = String(e); });
  }
  function range(arr) { return [Math.min.apply(null, arr), Math.max.apply(null, arr)]; }
  function render(o, cur) {
    var cond = o.condition, det = state.meta && state.meta.detector, att = state.meta && state.meta.attribution;
    $('capOrig').textContent = 'Original · ' + o.original_size[0] + '×' + o.original_size[1] + ' px';
    $('imgLaund').src = o.preview;
    $('capLaund').textContent = 'After "' + (CONDS.filter(function (c) { return c.id === cond; })[0].name) + '" · ' + o.laundered_size[0] + '×' + o.laundered_size[1] + ' px (shown ≤256 px)';
    var fr = $('imgLaund'); fr.onload = function () {
      var box = fr.getBoundingClientRect(), par = fr.parentElement.getBoundingClientRect();
      var c = $('cropBox'), side = Math.min(box.width, 224 * box.width / o.laundered_size[0]);
      c.style.width = side + 'px'; c.style.height = Math.min(box.height, 224 * box.height / o.laundered_size[1]) + 'px';
      c.style.left = (box.left - par.left + (box.width - side) / 2) + 'px'; c.style.top = (box.top - par.top + (box.height - parseFloat(c.style.height)) / 2) + 'px';
    };
    if (fr.complete) fr.onload();

    // --- truth
    var t = o.truth, tc = '';
    if (t) {
      tc = '<span class="chip neutral">True: ' + esc(t.true_label) + (t.kind === 'fake' ? ' (AI)' : ' (real)') + '</span>';
    } else tc = '<span class="chip neutral">Your upload: true source unknown</span>';
    $('truthChip').innerHTML = tc;

    // --- detector gauge
    var p = o.ai.prob, thr = o.ai.threshold_prob, flagged = o.ai.verdict === 'ai';
    $('gneedle').style.left = (p * 100) + '%'; $('gneedle').setAttribute('data-v', 'score ' + p.toFixed(2));
    $('gthr').style.left = (thr * 100) + '%'; $('gthr').setAttribute('data-v', thr.toFixed(2));
    $('gzone').style.left = (thr * 100) + '%';
    var chip = flagged ? '<span class="chip info">⚑ Flagged as AI-made</span>' : '<span class="chip neutral">○ Not flagged (looks real to this model)</span>';
    if (t) chip += ' ' + (t.detector_correct ? '<span class="chip good">✓ correct</span>' : '<span class="chip bad">✗ wrong</span>');
    $('aiChip').innerHTML = chip;
    var fprUnseen = state.find && state.find.unseen_fpr.sources.map(function (s) { return s.fpr; });
    var ins = det && det.in_source_heldout[cond];
    var tpr = ins && range(Object.keys(ins.tpr_by_generator).map(function (g) { return ins.tpr_by_generator[g].tpr; }));
    $('aiCaveat').innerHTML = '<b>Read with care.</b> The threshold is set so about 1% of real photos from the <i>training</i> collections are flagged (measured ' + (ins ? pct(ins.fpr_pooled) : '-') + ' on held-out photos at this laundering). On real-photo sources the detector had <i>never seen</i>, <b>' + (fprUnseen ? pct(Math.min.apply(null, fprUnseen)) + ' to ' + pct(Math.max.apply(null, fprUnseen)) : '0.5% to 17%') + '</b> of real photos were falsely flagged (leave-one-source-out, JPEG-75 + half size). It also misses fakes: on held-out test fakes it flags only ' + (tpr ? pct(tpr[0]) + ' to ' + pct(tpr[1]) : '-') + ' per generator. "Not flagged" is not proof of a real photo.';

    // --- generator bars
    var gens = o.generators, top = gens[0];
    $('gbars').innerHTML = gens.map(function (g, i) {
      return '<div class="gb' + (i === 0 ? ' top' : '') + (t && t.kind === 'fake' && g.name === t.true ? ' truth' : '') + '"><span class="nm">' + esc(g.label) + '</span><div class="track"><div class="fill" data-w="' + (g.prob * 100).toFixed(1) + '"></div></div><span class="pc">' + (g.prob * 100).toFixed(1) + '%</span></div>';
    }).join('');
    requestAnimationFrame(function () { requestAnimationFrame(function () { Array.prototype.forEach.call($('gbars').querySelectorAll('.fill'), function (f) { f.style.width = f.getAttribute('data-w') + '%'; }); }); });
    var gc = o.uncertain ? '<span class="chip warn">? Uncertain: top probability ' + (top.prob * 100).toFixed(0) + '% < ' + (o.msp_threshold * 100).toFixed(0) + '%</span>' : '<span class="chip info">Best guess: ' + esc(top.label) + '</span>';
    if (t && t.kind === 'fake') gc += ' ' + (t.generator_correct ? '<span class="chip good">✓ named correctly</span>' : '<span class="chip bad">✗ true source: ' + esc(t.true_label) + '</span>');
    if (t && t.kind === 'real') gc += ' <span class="chip neutral">real photo: any generator name is wrong</span>';
    $('genChip').innerHTML = gc;
    var hb = att && att.heldout_bacc_fold[cond], ref = att && att.reference_5fold.values[cond];
    var g2 = 'Held-out accuracy of this probe at this laundering: balanced accuracy <b>' + (hb != null ? hb.toFixed(2) : '-') + '</b> on ' + (att ? att.n_test : '-') + ' test fakes' + (ref != null ? ' (project-wide 5-fold figure: ' + ref.toFixed(2) + ')' : '') + '; chance is 0.125. ';
    g2 += 'The "uncertain" flag is a weak guard: an unknown generator is detected from the model\'s confidence at only <b>0.77 AUROC</b> (below the 0.80 bar), so images from unlisted generators are usually named as a known one.';
    if (!flagged) g2 = '<b>The detector did not flag this image.</b> Attribution is only meaningful after an image is judged AI-made; treat the bars as "closest known style". ' + g2;
    $('genCaveat').innerHTML = g2;
    if (t) { var k = state.cond; var ty = state.tally[k] || (state.tally[k] = { n: 0, ok: 0, seen: {} }); if (t.kind === 'fake' && !ty.seen[cur.sample.id]) { ty.seen[cur.sample.id] = 1; ty.n++; ty.ok += t.generator_correct ? 1 : 0; } if (ty.n) $('genCaveat').innerHTML += ' <span class="faint">Session at this laundering: ' + ty.ok + ' of ' + ty.n + ' AI samples named correctly.</span>'; }
    $('tech').innerHTML = [['Device', o.device], ['Prep (EXIF, fov512)', o.timing.prep_ms + ' ms'], ['Laundering', o.timing.launder_ms + ' ms'], ['DINOv2 forward', o.timing.forward_ms + ' ms'], ['Total', o.timing.total_ms + ' ms'], ['Detector logit', o.ai.logit.toFixed(3) + ' (threshold ' + o.ai.threshold_logit.toFixed(3) + ')'], ['Warnings', (o.warnings && o.warnings.length) ? o.warnings.join(' ') : 'none']].map(function (r) { return '<dt>' + r[0] + '</dt><dd>' + esc(r[1]) + '</dd>'; }).join('');
  }

  /* ------------------------------------------------------------ batch */
  $('batchBtn').onclick = function () {
    if (!state.samples.length) return;
    var cond = state.cond, res = [], i = 0, btn = $('batchBtn'); btn.disabled = true;
    function step() {
      if (i >= state.samples.length) { btn.disabled = false; return summarize(); }
      $('batchOut').innerHTML = '<span class="spin"></span> ' + (i + 1) + ' / ' + state.samples.length;
      var s = state.samples[i++];
      post({ type: 'sample', sample: s }, cond).then(function (o) { res.push({ s: s, o: o }); step(); }).catch(function () { btn.disabled = false; $('batchOut').textContent = 'failed'; });
    }
    function summarize() {
      var fk = res.filter(function (r) { return r.s.kind === 'fake' && r.o.ok; }), rl = res.filter(function (r) { return r.s.kind === 'real' && r.o.ok; });
      var gOk = fk.filter(function (r) { return r.o.truth.generator_correct; }).length, dOk = fk.filter(function (r) { return r.o.truth.detector_correct; }).length, rOk = rl.filter(function (r) { return r.o.truth.detector_correct; }).length;
      var unc = fk.filter(function (r) { return r.o.uncertain; }).length;
      var att = state.meta && state.meta.attribution, det = state.meta && state.meta.detector, ins = det && det.in_source_heldout[cond];
      var pooledTpr = ins ? (function () { var a = 0, n = 0; Object.keys(ins.tpr_by_generator).forEach(function (g) { if (g !== 'flux') { a += ins.tpr_by_generator[g].tpr * ins.tpr_by_generator[g].n; n += ins.tpr_by_generator[g].n; } }); return a / n; })() : null;
      var per = {}; fk.forEach(function (r) { var p = per[r.s.true] || (per[r.s.true] = [0, 0]); p[1]++; p[0] += r.o.truth.generator_correct ? 1 : 0; });
      $('batchOut').innerHTML = '<div class="scroll"><table><thead><tr><th>Check at "' + esc(cond) + '"</th><th class="num">Samples</th><th class="num">Held-out reference</th></tr></thead><tbody>' +
        '<tr><td>AI images named by the right generator</td><td class="num">' + gOk + ' / ' + fk.length + ' (' + (fk.length ? (gOk / fk.length * 100).toFixed(0) : '-') + '%)</td><td class="num">bacc ' + (att && att.heldout_bacc_fold[cond] != null ? att.heldout_bacc_fold[cond].toFixed(2) : '-') + '</td></tr>' +
        '<tr><td>AI images flagged by the detector</td><td class="num">' + dOk + ' / ' + fk.length + '</td><td class="num">' + (pooledTpr != null ? 'TPR ' + pct(pooledTpr) + ' (non-FLUX gens)' : '-') + '</td></tr>' +
        '<tr><td>Real photos left unflagged</td><td class="num">' + rOk + ' / ' + rl.length + '</td><td class="num">' + (ins ? 'FPR ' + pct(ins.fpr_pooled) + ' (seen sources)' : '-') + '</td></tr>' +
        '<tr><td>AI images marked "uncertain"</td><td class="num">' + unc + ' / ' + fk.length + '</td><td class="num">' + (att ? pct(att.uncertain_on_heldout[cond].frac_flagged) : '-') + '</td></tr></tbody></table></div>' +
        '<p class="tiny faint" style="margin-top:6px">Per generator (right / total): ' + Object.keys(per).map(function (g) { return esc(GEN[g]) + ' ' + per[g][0] + '/' + per[g][1]; }).join(' · ') + '. Only ' + fk.length + ' AI samples (about 3 per generator): expect ±10 percentage points of sampling noise. Samples are 768 px JPEG re-encodes of the originals, not the cached crops.</p>';
    }
    step();
  };

  /* ------------------------------------------------------------ findings */
  function figure(title, id, sentence, source) {
    var d = document.createElement('div'); d.className = 'card';
    d.innerHTML = '<h3>' + esc(title) + '</h3><p class="cap">' + esc(sentence) + '</p><div id="' + id + '" class="fig"></div><div class="src">Source: <code>' + esc(source) + '</code></div>';
    $('figs').appendChild(d); return $(id);
  }
  function labelClass(l) { var u = l.toUpperCase(); return u.indexOf('PASS') === 0 ? 'good' : u.indexOf('KILL') === 0 ? 'bad' : u.indexOf('NO') === 0 ? 'bad' : 'warn'; }
  function labelIcon(l) { var c = labelClass(l); return c === 'good' ? '✓ ' : c === 'bad' ? '✗ ' : '? '; }
  function renderFindings() {
    var F = state.find, sc = F.scorecard;
    $('scoreAttr').innerHTML = sc.attribution.map(function (h) {
      return '<div class="h"><div class="hd"><b>' + esc(h.id) + '</b><span class="chip ' + labelClass(h.label) + '">' + labelIcon(h.label) + esc(h.label) + '</span></div><div class="small muted">' + esc(h.method) + '</div><p style="margin:6px 0 4px">' + esc(h.plain) + '</p><div class="tiny faint">Key number: ' + esc(h.key_number) + ' · counts toward stop rule: ' + esc(h.counts) + ' · critic: ' + esc(h.critic) + '</div></div>';
    }).join('');
    $('scoreSrc').innerHTML = 'Source: <code>' + esc(sc.source) + '</code>';
    $('scoreDet').innerHTML = '<table><thead><tr><th>Experiment</th><th>Kind</th><th>Pre-registered label</th><th>Critic</th></tr></thead><tbody>' + sc.detection.map(function (r) { return '<tr><td>' + esc(r.experiment) + '</td><td>' + esc(r.kind) + '</td><td>' + esc(r.label) + '</td><td>' + esc(r.critic) + '</td></tr>'; }).join('') + '</tbody></table>';

    var L = F.laundering, nm = { 'nested / tmix': 'var(--s1)', 'nested / clean': 'var(--s2)', 'M1 / tmix': 'var(--s4)', 'M0 raw metadata': 'var(--neutral)' };
    var e = figure('Laundering: who survives?', 'fLaund', L.sentence, L.source);
    Charts.bars(e, { label: 'Balanced accuracy by laundering', cats: ['clean', 'jpeg75_resize05', 'social_sim', 'screenshot_sim*', 'mavali_medium*'], ymax: 1, refs: [{ y: L.chance, label: 'chance 0.125' }],
      series: Object.keys(L.series).map(function (k) { var s = L.series[k]; return { name: s.name, color: nm[k], values: s.values, lo: k === 'M0 raw metadata' ? null : s.lo, hi: k === 'M0 raw metadata' ? null : s.hi }; }) });
    e.insertAdjacentHTML('beforeend', '<p class="tiny faint">* never used in training. M0 is a property of the raw files (no laundering applies).</p>');

    var Y = F.layers;
    e = figure('Which DINOv2 layer carries the generator?', 'fLayers', Y.sentence + ' Contrast b8 − b12 at JPEG-75 + half size: T-mix ' + Y.contrast_b8_minus_b12.tmix + '; clean-train ' + Y.contrast_b8_minus_b12.clean + '.', Y.source);
    Charts.bars(e, { label: 'Balanced accuracy by layer', cats: ['block 4', 'block 8', 'block 12 (last)', 'concat (4+8+12)'], ymax: 1, refs: [{ y: Y.chance, label: 'chance 0.125' }],
      series: [{ name: 'trained on laundered copies', color: 'var(--s1)', values: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.tmix[b].v; }), lo: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.tmix[b].lo; }), hi: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.tmix[b].hi; }) },
        { name: 'trained on clean only', color: 'var(--s2)', values: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.clean[b].v; }), lo: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.clean[b].lo; }), hi: ['b4', 'b8', 'b12', 'cat'].map(function (b) { return Y.values.clean[b].hi; }) }] });

    var D = F.detector_embeddings;
    e = figure('Do real-vs-fake detectors know the generator?', 'fDet', D.sentence, D.source);
    e.innerHTML = '<div class="seg" id="detSeg" role="group" aria-label="Condition"></div><div id="detChart" style="margin-top:8px"></div>';
    function drawDet(ci) {
      $('detSeg').innerHTML = D.conditions.map(function (c, i) { return '<button aria-pressed="' + (i === ci) + '" data-i="' + i + '">' + c + '</button>'; }).join('');
      Array.prototype.forEach.call($('detSeg').querySelectorAll('button'), function (b) { b.onclick = function () { drawDet(+b.getAttribute('data-i')); }; });
      var rows = D.rows.filter(function (r) { return ['cat', 'commfor', 'bfree', 'clipdet', 'rine', 'b12'].indexOf(r.key) >= 0; }).map(function (r) { return { name: r.name, value: r.values[ci], bold: r.key === 'cat', color: r.key === 'cat' || r.key === 'b12' ? 'var(--s1)' : 'var(--s2)', key: r.key }; }).sort(function (a, b) { return b.value - a.value; });
      Charts.hbars($('detChart'), { label: 'Detector embeddings vs DINOv2', rows: rows, xmax: 1, refs: [{ x: D.chance, label: 'chance' }] });
    }
    drawDet(1);
    e.insertAdjacentHTML('beforeend', '<div class="legend"><span><i style="background:var(--s1)"></i>generic DINOv2 layers</span><span><i style="background:var(--s2)"></i>published real-vs-fake detectors (last layer)</span></div>');

    var LC = F.learning_curve;
    e = figure('Does a more varied training set of real photos fix false alarms?', 'fLC', LC.sentence, LC.source);
    e.innerHTML = '<div class="cols2"><div><h4>Mean AUROC on the held-out real source</h4><div id="lcA"></div></div><div><h4>False-positive rate at the 1% threshold</h4><div id="lcF"></div></div></div>';
    var ks = LC.mean.map(function (m) { return 'k=' + m.k; }), srcs = Object.keys(LC.per_source);
    Charts.lines($('lcA'), { label: 'AUROC vs number of training sources', xs: ks, xlabel: 'training sources', ymin: 0.5, ymax: 0.95, rightPad: 60,
      series: srcs.map(function (s) { return { name: s, color: 'var(--neutral)', faint: true, values: LC.per_source[s].auroc }; }).concat([{ name: 'mean of 6 held-out sources', color: 'var(--s1)', values: LC.mean.map(function (m) { return m.auroc; }), lo: LC.mean.map(function (m) { return m.auroc_lo; }), hi: LC.mean.map(function (m) { return m.auroc_hi; }), direct: 'mean' }]) });
    Charts.lines($('lcF'), { label: 'FPR vs number of training sources', xs: ks, xlabel: 'training sources', ymin: 0, ymax: 0.6, rightPad: 60, yfmt: function (v) { return Math.round(v * 100) + '%'; }, vfmt: pct,
      refs: [{ y: 0.01, label: '1%' }],
      series: srcs.map(function (s) { return { name: s, color: 'var(--neutral)', faint: true, values: LC.per_source[s].fpr }; }).concat([{ name: 'mean of 6 held-out sources', color: 'var(--s2)', values: LC.mean.map(function (m) { return m.fpr; }), lo: LC.mean.map(function (m) { return m.fpr_lo; }), hi: LC.mean.map(function (m) { return m.fpr_hi; }), direct: 'mean' }]) });

    var U = F.unseen_fpr;
    var nice = { VISION: 'VISION (2010s phones)', RAISE: 'RAISE (DSLR)', OI: 'Open Images', UNS: 'Unsplash', WMC: 'Wikimedia Commons', HDRP: 'HDR+ (phone finals)' };
    e = figure('False alarms on real photos the detector never saw', 'fUnseen', U.sentence, U.source);
    Charts.hbars(e, { label: 'FPR by held-out real source', xmax: 0.3, xticks: 6, fmt: pct, refs: [{ x: 0.01, label: '1% target' }],
      rows: U.sources.slice().sort(function (a, b) { return a.fpr - b.fpr; }).map(function (s) { return { name: nice[s.source] || s.source, value: s.fpr, lo: s.cp_lo, hi: s.cp_hi, color: 'var(--s1)', note: s.k + ' of ' + s.n + ' real photos flagged; AUROC ' + s.auroc.toFixed(2) }; }) });
    e.insertAdjacentHTML('beforeend', '<p class="tiny faint">Whiskers: Clopper-Pearson 95% interval. E7 (two training sources) vs E10 (five), same condition: ' + Object.keys(U.e7_vs_e10).map(function (k) { return esc(nice[k] || k) + ' ' + esc(U.e7_vs_e10[k].fpr_e7) + ' → ' + esc(U.e7_vs_e10[k].fpr_k5); }).join('; ') + '.</p>');

    var O = F.open_set;
    e = figure('Can the probe say "none of the above"?', 'fOpen', O.sentence, O.source);
    Charts.hbars(e, { label: 'Open-set AUROC per held-out generator', xmin: 0.5, xmax: 1, fmt: f3, refs: [{ x: 0.8, label: 'bar 0.80' }],
      rows: O.rows.map(function (r) { return { name: GEN[r.generator] || r.generator, value: r.msp, lo: r.msp_lo, hi: r.msp_hi, color: 'var(--s1)' }; }) });
    e.insertAdjacentHTML('beforeend', '<p class="tiny faint">Mean ' + O.mean_msp.toFixed(3) + '. A distance-based score (Mahalanobis) is at chance level, 0.55.</p>');

    var N = F.near_twins;
    e = figure('Near-twins: Stable Diffusion 1.3 vs 1.4', 'fTwin', N.sentence, N.source);
    Charts.bars(e, { label: 'SD1.3 vs SD1.4 balanced accuracy', cats: N.rows.map(function (r) { return r.condition; }), ymin: 0.4, ymax: 0.8, yticks: 4, refs: [{ y: 0.5, label: 'chance 0.5' }, { y: N.null_p95.jpeg75_resize05, label: 'shuffled-label p95 0.54' }],
      series: [{ name: 'probe (block 8, T-mix)', color: 'var(--s1)', values: N.rows.map(function (r) { return r.bacc; }), lo: N.rows.map(function (r) { return r.lo; }), hi: N.rows.map(function (r) { return r.hi; }) }] });

    if (F.live_model) {
      var M = F.live_model, cs = ['clean', 'jpeg75_resize05', 'social_sim', 'screenshot_sim', 'mavali_medium'];
      e = figure('This demo\'s own attribution model', 'fLive', 'The model behind the Analyze tab was trained on 4 of the 5 folds of the AI images; the bars compare its balanced accuracy on the held-out fold (' + M.n_test + ' images) with the project-wide 5-fold figure.', M.source + '; ' + 'experiments/020-attr-foundation/results/tables.md');
      Charts.bars(e, { label: 'Demo model vs 5-fold', cats: cs, ymax: 1, refs: [{ y: 0.125, label: 'chance' }], valueLabels: true,
        series: [{ name: 'this demo (held-out fold ' + M.fold + ')', color: 'var(--s1)', values: cs.map(function (c) { return M.heldout_bacc[c]; }) }, { name: 'project 5-fold (020)', color: 'var(--s4)', values: cs.map(function (c) { return M.reference_5fold.values[c]; }) }] });
    }
  }

  /* ------------------------------------------------------------ phone fingerprints */
  var prnu = { arm: 'nat', met: 'median' };
  function setupPhone() {
    var P = state.find.prnu;
    $('armSeg').innerHTML = Object.keys(P.arms).map(function (a) { return '<button data-a="' + a + '" aria-pressed="' + (a === prnu.arm) + '">' + esc(P.arms[a].name) + '</button>'; }).join('');
    $('metSeg').innerHTML = '<button data-m="median" aria-pressed="true">median PCE (log colour)</button><button data-m="share" aria-pressed="false">share of photos with PCE ≥ 60</button>';
    Array.prototype.forEach.call($('armSeg').querySelectorAll('button'), function (b) { b.onclick = function () { prnu.arm = b.getAttribute('data-a'); renderHeat(); }; });
    Array.prototype.forEach.call($('metSeg').querySelectorAll('button'), function (b) { b.onclick = function () { prnu.met = b.getAttribute('data-m'); renderHeat(); }; });
    $('prnuSrc').innerHTML = 'Source: <code>' + esc(P.source) + '</code>. ' + esc(P.sentence);
  }
  function hex2rgb(h) { h = h.trim().replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function heatColor(t) {
    var cs = getComputedStyle(document.documentElement), a = hex2rgb(cs.getPropertyValue('--heat0')), b = hex2rgb(cs.getPropertyValue('--heat1'));
    t = Math.max(0, Math.min(1, t)); var c = a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); });
    return { css: 'rgb(' + c.join(',') + ')', lum: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255 };
  }
  function renderHeat() {
    if (!state.find || !$('p-phone').classList.contains('active')) return;
    var P = state.find.prnu, A = P.arms[prnu.arm], dev = P.devices, n = dev.length, cell = 38, ml = 52, mt = 56, W = ml + n * cell + 8, H = mt + n * cell + 8;
    Array.prototype.forEach.call($('armSeg').querySelectorAll('button'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-a') === prnu.arm); });
    Array.prototype.forEach.call($('metSeg').querySelectorAll('button'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-m') === prnu.met); });
    var LO = -1, HI = 4; // log10(PCE) from 0.1 to 10000
    function tOf(v) { return prnu.met === 'median' ? (Math.log10(Math.max(0.1, Math.min(10000, v))) - LO) / (HI - LO) : v; }
    var s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="PCE heatmap, ' + n + ' by ' + n + ' devices">';
    var gi = 0, off = 0;
    P.groups.forEach(function (g) { var x = ml + off * cell; s += '<text x="' + (x + g[1].length * cell / 2) + '" y="' + (mt - 30) + '" text-anchor="middle" style="font-size:10.5px">' + esc(g[0].replace('other single-device models', 'single-device models').replace('iPhone ', '')) + '</text><line class="axisline" x1="' + (x + 3) + '" x2="' + (x + g[1].length * cell - 3) + '" y1="' + (mt - 24) + '" y2="' + (mt - 24) + '"/>'; off += g[1].length; });
    dev.forEach(function (d, j) { s += '<text x="' + (ml + j * cell + cell / 2) + '" y="' + (mt - 8) + '" text-anchor="middle" style="font-size:10.5px">' + d + '</text><text x="' + (ml - 6) + '" y="' + (mt + j * cell + cell / 2 + 4) + '" text-anchor="end" style="font-size:10.5px">' + d + '</text>'; });
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var v = prnu.met === 'median' ? A.median_pce[i][j] : A.share_ge60[i][j], c = heatColor(tOf(v));
      var lab = prnu.met === 'median' ? (v >= 100 ? Math.round(v) : v.toFixed(1)) : (v * 100).toFixed(0);
      var tp = '<b>Photo from ' + dev[i] + ', fingerprint of ' + dev[j] + '</b>' + (i === j ? ' (same device)' : '') + '<br>median PCE ' + A.median_pce[i][j].toFixed(1) + ' · max ' + A.max_pce[i][j].toFixed(1) + '<br>' + (A.share_ge60[i][j] * 100).toFixed(0) + '% of ' + A.n_photos[dev[i]] + ' photos have PCE ≥ 60';
      s += '<g class="mk" tabindex="0" data-tip="' + esc(tp) + '"><rect x="' + (ml + j * cell + 1) + '" y="' + (mt + i * cell + 1) + '" width="' + (cell - 2) + '" height="' + (cell - 2) + '" rx="4" style="fill:' + c.css + '"' + (i === j ? ' stroke="var(--ink)" stroke-width="1.2"' : '') + '/><text x="' + (ml + j * cell + cell / 2) + '" y="' + (mt + i * cell + cell / 2 + 3.5) + '" text-anchor="middle" style="font-size:' + (n > 12 ? 9.5 : 11) + 'px;fill:' + (c.lum < 0.55 ? '#fff' : '#111') + '">' + lab + '</text></g>';
    }
    s += '<text x="' + (ml + n * cell / 2) + '" y="' + (H - 0) + '" text-anchor="middle" style="display:none"></text></svg>';
    $('heat').innerHTML = s; Charts.bindTips($('heat'));
    // legend
    var stops = []; for (var k = 0; k <= 10; k++) stops.push(heatColor(k / 10).css);
    var marks = prnu.met === 'median' ? [0.1, 1, 10, 60, 100, 1000, 10000] : [0, 0.25, 0.5, 0.75, 1];
    $('heatLegend').innerHTML = '<div style="margin:10px 0 2px;max-width:520px"><div style="height:12px;border-radius:6px;background:linear-gradient(90deg,' + stops.join(',') + ')"></div><div style="position:relative;height:26px">' + marks.map(function (m) { var t = prnu.met === 'median' ? tOf(m) : m; return '<span style="position:absolute;left:' + (t * 100) + '%;transform:translateX(-50%);font-size:11px;color:' + (m === 60 ? 'var(--s2)' : 'var(--ink-3)') + ';font-weight:' + (m === 60 ? 700 : 400) + '">' + (m === 60 ? '60 ▲' : prnu.met === 'median' ? m : Math.round(m * 100) + '%') + '</span>'; }).join('') + '</div><div class="tiny faint">' + (prnu.met === 'median' ? 'PCE, log scale (60 = conventional match line)' : 'share of photos with PCE ≥ 60') + '</div></div>';
    var st = [[pct(A.top1), 'top-1 device accuracy (chance 7.1%)'], [Math.round(A.median_true).toLocaleString() + ' vs ' + A.median_false.toFixed(1), 'median PCE: right device vs wrong'], [pct(A.tpr60), 'right-device photos with PCE ≥ 60'], [A.false_ge60 + ' of ' + A.n_false_pairs.toLocaleString(), 'wrong-device pairs with PCE ≥ 60 (max ' + A.max_false.toFixed(1) + ')']];
    $('prnuStats').innerHTML = st.map(function (x) { return '<div class="s"><div class="v">' + x[0] + '</div><div class="k">' + esc(x[1]) + '</div></div>'; }).join('');
    $('prnuNote').innerHTML = (prnu.arm === 'nat' ? 'On untouched files the right phone wins almost every time, and even phones of the same model (cells inside one group, e.g. the three 5c phones) score only about 3 to 15 against their own hundreds or thousands. ' : 'After the platform\'s re-encode, the right phone still usually has the highest score, but the score falls below the 60 match line for most photos: choosing the best of 14 needs no threshold, a yes/no match does. ') + 'Arm: ' + esc(A.desc) + ', ' + A.n_photos_total + ' photos.';
    $('prnuCav').innerHTML = '<b>Honest limits.</b> A textbook replication on 14 same-size 2010s iPhones (4s to 6), with 13 to 14 natural photos per device. It needs reference photos from the claimed device, so it verifies a claim rather than screening images. "No false hit" is a bound (about 0.12% from 2,561 wrong pairs), not zero: the largest wrong-device PCE reached ' + A.max_false.toFixed(1) + '. Modern neural-ISP phones are untested.';
  }

  /* ------------------------------------------------------------ caveats */
  function renderCaveats() {
    var C = [
      ['Eight known generators only', 'The attribution probe knows DALL-E 2/3, Firefly, Midjourney v5, SD 2, SDXL, SD 3 and FLUX. It must pick one of them for every image, including real photos and generators it has never seen.', 'docs/10 section 4 (H-A1); experiments/020-attr-foundation/results/tables.md'],
      ['Near-twins are inseparable', 'Stable Diffusion 1.3 vs 1.4 sits at chance (best 0.527, inside the shuffled-label null). Report names at family level, not version.', 'experiments/020-attr-foundation/secondary/tables.md'],
      ['Open-set is unreliable', 'Unknown generators are mapped to known ones (SD 1.x → SD2/Midjourney/SD3; GLIDE → DALL-E 2 91% of the time, partly an upscaling confound). The "uncertain" flag reaches only 0.77 AUROC, below the 0.80 bar.', 'docs/10 sections 4 and 7'],
      ['The detector is not a verdict', 'At its 1% threshold it flags only a fraction of fakes (FLUX the least), and on real-photo sources it never saw, 0.5% to 17% of real photos were flagged. The "FPR ≤ 1% on fresh phone photos" target is untested on modern phones.', 'experiments/019-source-diversity/results/tables.md; docs/03 addenda'],
      ['2010s phones only', 'The fingerprint results use VISION (iPhone 4s to 6). Neural-ISP phones and current WhatsApp/Facebook pipelines are untested; Samsung default-mode fingerprints show high false-positive rates in the literature.', 'docs/10 sections 7 and 8'],
      ['Small samples', 'About 24 test images per generator per fold; 204 test images behind this demo; 3 samples per generator on the Analyze tab; 14 devices in the PRNU study. Hit rates on a few samples are noisy (±10 points or more).', 'docs/10 section 8'],
      ['Simulated launderings', 'JPEG, resize and screenshot effects are simulated by code, not real uploads. Real platform pipelines were not tested with fakes.', 'docs/10 section 8'],
      ['Style is version-specific', 'The cue is mid-level rendering style. A new version, fine-tune, LoRA or style preset may be misattributed until it is collected. This update lag was not measured.', 'docs/10 section 7; experiments/020-attr-foundation/results/critique.md'],
      ['Attribution is not provenance', 'File metadata alone names the generator at 0.69 on raw files, and a pristine VAE-decoder match reaches 0.97. Laundering destroys both. Content credentials (C2PA) and watermarks were not studied.', 'docs/10 sections 3, 4, 7'],
      ['Licences', 'Training data include CC BY-NC-SA (Synthbuster), GRIP non-commercial (B-Free, RAISE) and other terms. The models inherit the non-commercial restriction; sample images are not redistributed.', 'docs/10 section 8; kb/datasets'],
      ['One backbone, one fold', 'A single frozen DINOv2-base with linear probes. The demo model trains on 4 of 5 folds so that its samples stay honest; the published numbers use cross-validation over all 1,020 fakes.', 'experiments/020-attr-foundation/config.json'],
      ['Research demo, not a product', 'Uploads smaller than 512 px are upscaled (outside the validated pipeline); sample thumbnails are re-encoded JPEGs. Never use these scores to accuse a person or an image.', 'poc/fingerprint-demo/README.md']
    ];
    $('cavs').innerHTML = C.map(function (c, i) { return '<div class="c"><h3><span class="ic">' + (i + 1) + '</span>' + esc(c[0]) + '</h3><p class="muted">' + esc(c[1]) + '</p><div class="src">Source: <code>' + esc(c[2]) + '</code></div></div>'; }).join('');
  }

  boot();
})();
