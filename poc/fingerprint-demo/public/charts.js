/* Tiny inline-SVG chart helpers (no libraries). Colours come from CSS variables so light/dark follow the page. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function f3(v) { return v == null || isNaN(v) ? '-' : (Math.round(v * 1000) / 1000).toFixed(3); }
  function pct(v) { return v == null ? '-' : (v * 100).toFixed(1) + '%'; }
  var tip = null;
  function tipEl() { return tip || (tip = document.getElementById('tip')); }
  function showTip(ev, html) {
    var t = tipEl(); t.innerHTML = html; t.classList.add('on');
    var x = ev.clientX + 14, y = ev.clientY + 14, w = t.offsetWidth, h = t.offsetHeight;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    t.style.left = x + 'px'; t.style.top = y + 'px';
  }
  function hideTip() { tipEl().classList.remove('on'); }
  function bindTips(root) {
    var els = root.querySelectorAll('[data-tip]');
    Array.prototype.forEach.call(els, function (el) {
      el.addEventListener('mousemove', function (e) { showTip(e, el.getAttribute('data-tip')); });
      el.addEventListener('mouseleave', hideTip);
      el.addEventListener('focus', function () { var r = el.getBoundingClientRect(); showTip({ clientX: r.left + r.width / 2, clientY: r.top }, el.getAttribute('data-tip')); });
      el.addEventListener('blur', hideTip);
    });
  }
  function legend(series) {
    return '<div class="legend">' + series.map(function (s) { return '<span><i style="background:' + s.color + '"></i>' + esc(s.name) + '</span>'; }).join('') + '</div>';
  }
  function table(headers, rows) {
    return '<details class="tbl"><summary>Table view</summary><div class="scroll"><table><thead><tr>' +
      headers.map(function (h, i) { return '<th' + (i ? ' class="num"' : '') + '>' + esc(h) + '</th>'; }).join('') + '</tr></thead><tbody>' +
      rows.map(function (r) { return '<tr>' + r.map(function (c, i) { return '<td' + (i ? ' class="num"' : '') + '>' + esc(c) + '</td>'; }).join('') + '</tr>'; }).join('') +
      '</tbody></table></div></details>';
  }
  function ticks(lo, hi, n) { var out = [], st = (hi - lo) / n; for (var i = 0; i <= n; i++) out.push(lo + st * i); return out; }

  /* grouped vertical bars. spec: {cats, series:[{name,color,values,lo,hi}], ymin,ymax,yticks, fmt, refs:[{y,label}], h, ylabel} */
  function bars(el, spec) {
    var W = 720, H = spec.h || 280, m = { l: 44, r: 74, t: 12, b: 44 };
    var y0 = spec.ymin || 0, y1 = spec.ymax || 1, fmt = spec.fmt || f3;
    var pw = W - m.l - m.r, ph = H - m.t - m.b, nc = spec.cats.length, ns = spec.series.length;
    var gw = pw / nc, bw = Math.min(46, (gw * 0.78 - 2 * (ns - 1)) / ns);
    var Y = function (v) { return m.t + ph - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0) * ph; };
    var s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(spec.label || 'chart') + '">';
    ticks(y0, y1, spec.yticks || 5).forEach(function (t) {
      s += '<line class="gridl" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + (Math.round(t * 100) / 100) + '</text>';
    });
    s += '<line class="axisline" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(y0) + '" y2="' + Y(y0) + '"/>';
    spec.cats.forEach(function (c, i) {
      var cx = m.l + gw * i + gw / 2, x0 = cx - (ns * bw + (ns - 1) * 2) / 2;
      spec.series.forEach(function (sr, j) {
        var v = sr.values[i]; if (v == null) return;
        var x = x0 + j * (bw + 2), y = Y(v), h = Math.max(1, Y(y0) - y);
        var tp = '<b>' + esc(c) + '</b><br>' + esc(sr.name) + ': <b>' + fmt(v) + '</b>' + (sr.lo && sr.lo[i] != null ? '<br>95% CI [' + fmt(sr.lo[i]) + ', ' + fmt(sr.hi[i]) + ']' : '');
        s += '<path class="mk" tabindex="0" data-tip="' + esc(tp) + '" style="fill:' + sr.color + '" d="M' + x + ',' + (y + h) + 'V' + (y + Math.min(4, h)) + 'Q' + x + ',' + y + ' ' + (x + Math.min(4, bw / 2)) + ',' + y + 'H' + (x + bw - Math.min(4, bw / 2)) + 'Q' + (x + bw) + ',' + y + ' ' + (x + bw) + ',' + (y + Math.min(4, h)) + 'V' + (y + h) + 'Z"/>';
        if (sr.lo && sr.lo[i] != null) s += '<line style="stroke:var(--ink)" stroke-width="1.2" x1="' + (x + bw / 2) + '" x2="' + (x + bw / 2) + '" y1="' + Y(sr.lo[i]) + '" y2="' + Y(sr.hi[i]) + '"/>';
        if (spec.valueLabels) s += '<text x="' + (x + bw / 2) + '" y="' + (y - 5) + '" text-anchor="middle" style="font-size:10.5px">' + (Math.round(v * 100) / 100) + '</text>';
      });
      s += '<text x="' + cx + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(c) + '</text>';
    });
    (spec.refs || []).forEach(function (r) {
      s += '<line class="ref" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '"/><text x="' + (W - m.r) + '" y="' + (Y(r.y) - 4) + '" text-anchor="end" style="fill:var(--ink-3)">' + esc(r.label) + '</text>';
    });
    s += '</svg>';
    el.innerHTML = (spec.series.length > 1 ? legend(spec.series) : '') + s +
      table(['Category'].concat(spec.series.map(function (x) { return x.name; })), spec.cats.map(function (c, i) { return [c].concat(spec.series.map(function (x) { return x.values[i] == null ? '-' : fmt(x.values[i]) + (x.lo && x.lo[i] != null ? ' [' + fmt(x.lo[i]) + ', ' + fmt(x.hi[i]) + ']' : ''); })); }));
    bindTips(el);
  }

  /* horizontal bars. spec: {rows:[{name,value,lo,hi,color,note}], xmax, fmt, refs:[{x,label}]} */
  function hbars(el, spec) {
    var rowH = 26, W = 720, m = { l: 190, r: 56, t: 8, b: 26 }, n = spec.rows.length, H = m.t + m.b + n * rowH;
    var x0 = spec.xmin || 0, x1 = spec.xmax || 1, fmt = spec.fmt || f3, pw = W - m.l - m.r;
    var X = function (v) { return m.l + (Math.max(x0, Math.min(x1, v)) - x0) / (x1 - x0) * pw; };
    var s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(spec.label || 'chart') + '">';
    ticks(x0, x1, spec.xticks || 5).forEach(function (t) { s += '<line class="gridl" x1="' + X(t) + '" x2="' + X(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(t) + '" y="' + (H - 8) + '" text-anchor="middle">' + (Math.round(t * 100) / 100) + '</text>'; });
    spec.rows.forEach(function (r, i) {
      var y = m.t + i * rowH + 5, h = rowH - 10;
      var tp = '<b>' + esc(r.name) + '</b><br>' + fmt(r.value) + (r.lo != null ? '<br>95% interval [' + fmt(r.lo) + ', ' + fmt(r.hi) + ']' : '') + (r.note ? '<br>' + esc(r.note) : '');
      s += '<text x="' + (m.l - 8) + '" y="' + (y + h - 3) + '" text-anchor="end" style="' + (r.bold ? 'font-weight:600;fill:var(--ink)' : '') + '">' + esc(r.name) + '</text>';
      s += '<path class="mk" tabindex="0" data-tip="' + esc(tp) + '" style="fill:' + (r.color || 'var(--s1)') + '" d="M' + m.l + ',' + y + 'H' + (X(r.value) - 4) + 'Q' + X(r.value) + ',' + y + ' ' + X(r.value) + ',' + (y + 4) + 'V' + (y + h - 4) + 'Q' + X(r.value) + ',' + (y + h) + ' ' + (X(r.value) - 4) + ',' + (y + h) + 'H' + m.l + 'Z"/>';
      if (r.lo != null) s += '<line style="stroke:var(--ink)" stroke-width="1.2" y1="' + (y + h / 2) + '" y2="' + (y + h / 2) + '" x1="' + X(r.lo) + '" x2="' + X(r.hi) + '"/>';
      s += '<text x="' + (Math.max(X(r.value), r.hi != null ? X(r.hi) : 0) + 6) + '" y="' + (y + h - 3) + '">' + fmt(r.value) + '</text>';
    });
    (spec.refs || []).forEach(function (r) { s += '<line class="ref" x1="' + X(r.x) + '" x2="' + X(r.x) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(r.x) + '" y="' + (m.t - 0) + '" text-anchor="middle" dy="-0" style="fill:var(--ink-3)">' + esc(r.label) + '</text>'; });
    s += '</svg>';
    el.innerHTML = s + table(['Row', 'Value'], spec.rows.map(function (r) { return [r.name, fmt(r.value) + (r.lo != null ? ' [' + fmt(r.lo) + ', ' + fmt(r.hi) + ']' : '')]; }));
    bindTips(el);
  }

  /* lines. spec: {xs, series:[{name,color,values,lo,hi,faint,direct}], ymin,ymax, xlabel, fmt, refs, xfmt} */
  function lines(el, spec) {
    var W = 520, H = spec.h || 260, m = { l: 44, r: spec.rightPad || 70, t: 12, b: 38 }, fmt = spec.fmt || f3;
    var y0 = spec.ymin || 0, y1 = spec.ymax || 1, xs = spec.xs, pw = W - m.l - m.r, ph = H - m.t - m.b;
    var X = function (i) { return m.l + (xs.length === 1 ? 0 : i / (xs.length - 1)) * pw; };
    var Y = function (v) { return m.t + ph - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0) * ph; };
    var s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(spec.label || 'chart') + '">';
    ticks(y0, y1, 5).forEach(function (t) { s += '<line class="gridl" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + (spec.yfmt ? spec.yfmt(t) : Math.round(t * 100) / 100) + '</text>'; });
    xs.forEach(function (x, i) { s += '<text x="' + X(i) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(x) + '</text>'; });
    s += '<text x="' + (m.l + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(spec.xlabel || '') + '</text>';
    (spec.refs || []).forEach(function (r) { s += '<line class="ref" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '"/><text x="' + (W - m.r + 4) + '" y="' + (Y(r.y) + 4) + '" style="fill:var(--ink-3)">' + esc(r.label) + '</text>'; });
    spec.series.forEach(function (sr) {
      if (sr.lo) {
        var up = sr.hi.map(function (v, i) { return X(i) + ',' + Y(v); }), dn = sr.lo.map(function (v, i) { return X(i) + ',' + Y(v); }).reverse();
        s += '<polygon points="' + up.concat(dn).join(' ') + '" style="fill:' + sr.color + ';opacity:.16"/>';
      }
      s += '<polyline fill="none" style="stroke:' + sr.color + ';opacity:' + (sr.faint ? 0.45 : 1) + '" stroke-width="' + (sr.faint ? 1.2 : 2.2) + '" stroke-linejoin="round" points="' + sr.values.map(function (v, i) { return X(i) + ',' + Y(v); }).join(' ') + '"/>';
      sr.values.forEach(function (v, i) {
        var tp = '<b>' + esc(sr.name) + '</b><br>' + esc(spec.xlabel || 'x') + ' ' + esc(xs[i]) + ': <b>' + (spec.vfmt || fmt)(v) + '</b>' + (sr.lo ? '<br>95% CI [' + (spec.vfmt || fmt)(sr.lo[i]) + ', ' + (spec.vfmt || fmt)(sr.hi[i]) + ']' : '');
        s += '<circle class="mk" tabindex="0" data-tip="' + esc(tp) + '" cx="' + X(i) + '" cy="' + Y(v) + '" r="' + (sr.faint ? 3 : 4.5) + '" style="fill:' + sr.color + ';stroke:var(--surface)" stroke-width="2"/>';
      });
      if (sr.direct) s += '<text x="' + (X(xs.length - 1) + 8) + '" y="' + (Y(sr.values[sr.values.length - 1]) + 4) + '" style="font-size:11px;fill:var(--ink-2)">' + esc(sr.direct) + '</text>';
    });
    s += '</svg>';
    el.innerHTML = s + table(['x'].concat(spec.series.map(function (x) { return x.name; })), xs.map(function (x, i) { return [x].concat(spec.series.map(function (sr) { return (spec.vfmt || fmt)(sr.values[i]); })); }));
    bindTips(el);
  }

  window.Charts = { bars: bars, hbars: hbars, lines: lines, legend: legend, esc: esc, f3: f3, pct: pct, showTip: showTip, hideTip: hideTip, bindTips: bindTips, table: table };
})();
