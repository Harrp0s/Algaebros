/* Algaebro dashboard. No build step: this file plus MapLibre GL (bundled next to it) is the whole front end. */
(function () {
  'use strict';

  // ====================================================================== helpers
  const $ = (s) => document.querySelector(s);
  const esc = (t) => String(t == null ? '' : t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const COLORS = { high: '#e0472d', elevated: '#e29a14', low: '#25c08a', unknown: '#98a2b3' };
  const WORDS = { high: 'High chlorophyll reading (possible algae)', elevated: 'Watch', low: 'Nothing unusual in the latest reading', unknown: 'No reading right now' };
  const DAY = 86400000;

  const prefs = {
    get(k, d) { try { const v = localStorage.getItem('algaebro.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('algaebro.' + k, JSON.stringify(v)); } catch (e) { /* private mode: ignore */ } },
  };

  async function api(path, body) {
    const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    let r;
    try { r = await fetch(path, opt); } catch (e) { return { ok: false, status: 0, data: { error: 'Could not reach the Algaebro server.' } }; }
    let data = {};
    try { data = await r.json(); } catch (e) { /* not json */ }
    return { ok: r.ok, status: r.status, data };
  }

  let toastTimer;
  function toast(msg, ms) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), ms || 4200);
  }

  function circlePoly(lon, lat, rM, n) {
    n = n || 40;
    const ring = [];
    const dLat = rM / 110574;
    const dLon = rM / (111320 * Math.cos((lat * Math.PI) / 180));
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * 2 * Math.PI;
      ring.push([lon + dLon * Math.cos(a), lat + dLat * Math.sin(a)]);
    }
    return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [ring] } };
  }

  // ====================================================================== state
  let S = null;            // /api/state
  let FR = null;           // /api/frames
  let ZG = [];             // /api/zones (geometry)
  let heatMeta = null;
  let selected = null;
  let heatOn = prefs.get('heat', true);
  let ahead = 0;                                      // 0 = past or now; 1-7 = days ahead of the latest reading
  let view = prefs.get('view', 'rule');               // 'rule' = latest reading carried forward, 'ml' = ML model
  const hotMarkers = {};
  let selMarker = null, pinMarker = null, popup = null, pinDragging = false;
  let styleLoaded = false, usingFallback = false;

  // ====================================================================== map
  const STYLE_URL = 'https://tiles.openfreemap.org/styles/positron';
  const FALLBACK_STYLE = {
    version: 8,
    sources: {
      osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 19, attribution: '&copy; OpenStreetMap contributors' },
    },
    layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
  };
  const BLANK_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

  const map = new maplibregl.Map({
    container: 'map', style: STYLE_URL, center: [-71.105, 42.357], zoom: 12.6, pitch: prefs.get('pitch', 52), bearing: -18,
    maxPitch: 75, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right');

  function useFallbackStyle() {
    if (usingFallback) return;
    usingFallback = true;
    toast('The detailed base map could not load, so a simpler one is shown.', 5200);
    map.setStyle(FALLBACK_STYLE);
  }
  map.on('error', (e) => {
    if (styleLoaded) return;                        // tile hiccups after the style is up are not worth reacting to
    const msg = (e && e.error && e.error.message) || '';
    if (/style|fetch|failed|network|load|40\d|50\d/i.test(msg)) useFallbackStyle();
  });
  setTimeout(() => { if (!styleLoaded) useFallbackStyle(); }, 9000);

  function firstSymbolId() {
    const layers = (map.getStyle() || {}).layers || [];
    const l = layers.find((x) => x.type === 'symbol');
    return l ? l.id : undefined;
  }

  function installLayers() {
    if (map.getSource('heat')) return;             // already installed for this style
    const before = firstSymbolId();
    // 3D buildings, when the base map has OpenMapTiles building data
    const style = map.getStyle();
    const vec = Object.entries(style.sources || {}).find(([, v]) => v.type === 'vector');
    if (vec && (style.layers || []).some((l) => l['source-layer'] === 'building') && !map.getLayer('buildings-3d')) {
      map.addLayer({
        id: 'buildings-3d', type: 'fill-extrusion', source: vec[0], 'source-layer': 'building', minzoom: 13,
        paint: {
          'fill-extrusion-color': '#d9e3de',
          'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 6],
          'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
          'fill-extrusion-opacity': 0.82,
        },
      }, before);
    }
    const c = heatMeta ? heatMeta.coordinates : [[-71.14, 42.38], [-71.07, 42.38], [-71.07, 42.34], [-71.14, 42.34]];
    map.addSource('heat', { type: 'image', url: heatMeta ? heatMeta.url : BLANK_PNG, coordinates: c });
    map.addLayer({ id: 'heat-layer', type: 'raster', source: 'heat', paint: { 'raster-opacity': heatOn ? 0.93 : 0, 'raster-fade-duration': 0, 'raster-resampling': 'linear' } }, before);

    map.addSource('hot', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'hot-columns', type: 'fill-extrusion', source: 'hot', paint: {
      'fill-extrusion-color': '#e0472d', 'fill-extrusion-height': ['get', 'h'], 'fill-extrusion-base': 0, 'fill-extrusion-opacity': 0.5 } });
    map.addSource('pin-ring', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'pin-ring-fill', type: 'fill', source: 'pin-ring', paint: { 'fill-color': '#25c08a', 'fill-opacity': 0.14 } });
    map.addLayer({ id: 'pin-ring-line', type: 'line', source: 'pin-ring', paint: { 'line-color': '#16a574', 'line-width': 2, 'line-dasharray': [2, 2] } });

    // invisible dots over every water pixel sample: they make the river clickable and hoverable
    map.addSource('zone-pts', { type: 'geojson', data: pointsGeojson() });
    map.addLayer({ id: 'zone-hit', type: 'circle', source: 'zone-pts', paint: { 'circle-radius': 12, 'circle-opacity': 0 } });
    syncHot();
    if (S && S.demo.enabled) syncPinRing();
  }

  map.on('style.load', () => { styleLoaded = true; installLayers(); });
  map.on('click', (e) => {
    const hit = map.getLayer('zone-hit') ? map.queryRenderedFeatures(e.point, { layers: ['zone-hit'] })[0] : null;
    if (hit) select(hit.properties.id);
    if (S && S.demo.enabled && pinMarker) { pinMarker.setLngLat(e.lngLat); syncPinRing(); sendPin(e.lngLat); }
  });
  map.on('mousemove', 'zone-hit', (e) => {
    map.getCanvas().style.cursor = 'pointer';
    const f = e.features && e.features[0];
    if (!f) return;
    if (!popup) popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12 });
    popup.setLngLat(e.lngLat).setText(f.properties.name).addTo(map);
  });
  map.on('mouseleave', 'zone-hit', () => { map.getCanvas().style.cursor = ''; if (popup) popup.remove(); });

  function pointsGeojson() {
    const feats = [];
    ZG.forEach((z) => z.points.forEach((p) => feats.push({ type: 'Feature', properties: { id: z.id, name: z.name }, geometry: { type: 'Point', coordinates: [p[0], p[1]] } })));
    return { type: 'FeatureCollection', features: feats };
  }

  function mapPadding() {
    const panelOpen = !$('#panel').classList.contains('hidden');
    return { top: 110, bottom: 190, left: 300, right: panelOpen ? 420 : 70 };
  }
  function fitRiver(animate) {
    const pts = [];
    ZG.forEach((z) => pts.push([z.anchor[1], z.anchor[0]]));
    if (!pts.length) return;
    const b = pts.reduce((bb, p) => bb.extend(p), new maplibregl.LngLatBounds(pts[0], pts[0]));
    map.fitBounds(b, { padding: mapPadding(), maxZoom: 15, duration: animate ? 900 : 0, pitch: prefs.get('pitch', 52), bearing: -18 });
  }

  // ====================================================================== heat overlay and legend
  let heatUrl = null;
  async function updateHeat() {
    const { ok, data } = await api('/api/heat?days=' + reqDays() + '&view=' + (mlView() ? 'ml' : 'rule'));
    if (!ok) return;
    if (data.url === heatUrl) return;
    heatUrl = data.url;
    heatMeta = data;
    const src = map.getSource('heat');
    if (src && src.updateImage) src.updateImage({ url: data.url, coordinates: data.coordinates });
    renderLegend();
  }

  function renderLegend() {
    if (!heatMeta) return;
    const lg = heatMeta.legend;
    const lo = lg.stops[0].value, hi = lg.stops[lg.stops.length - 1].value;
    $('#ramp').style.background = 'linear-gradient(90deg,' + lg.stops.map((s) => s.color + ' ' + (((s.value - lo) / (hi - lo)) * 100).toFixed(1) + '%').join(',') + ')';
    const pos = Math.max(0, Math.min(100, ((lg.high - lo) / (hi - lo)) * 100));
    $('#flagTick').style.left = pos + '%';
    $('#heatNote').textContent = heatMeta.note || '';
    $('#heatToggle').checked = heatOn;
  }
  $('#heatToggle').addEventListener('change', (e) => {
    heatOn = e.target.checked;
    prefs.set('heat', heatOn);
    if (map.getLayer('heat-layer')) map.setPaintProperty('heat-layer', 'raster-opacity', heatOn ? 0.93 : 0);
  });

  // ====================================================================== hotspots and selection
  function hotElement(z) {
    const el = document.createElement('div');
    el.className = 'hot';
    el.innerHTML = '<span class="ring"></span><span class="core">!</span><span class="chip"></span>';
    el.querySelector('.chip').textContent = z.name;
    el.title = z.name + ': high chlorophyll reading (possible algae)';
    el.addEventListener('click', (ev) => { ev.stopPropagation(); select(z.id, true); });
    return el;
  }

  function syncHot() {
    if (!S) return;
    const flagged = S.zones.filter((z) => z.flagged);
    document.body.classList.toggle('many-hot', flagged.length > 5);
    const ids = new Set(flagged.map((z) => z.id));
    Object.keys(hotMarkers).forEach((id) => { if (!ids.has(id)) { hotMarkers[id].remove(); delete hotMarkers[id]; } });
    flagged.forEach((z) => {
      if (!hotMarkers[z.id]) hotMarkers[z.id] = new maplibregl.Marker({ element: hotElement(z), anchor: 'center' }).setLngLat([z.anchor[1], z.anchor[0]]).addTo(map);
      hotMarkers[z.id].getElement().classList.toggle('selected', z.id === selected);
    });
    const src = map.getSource('hot');
    if (src) {
      src.setData({
        type: 'FeatureCollection',
        features: flagged.map((z) => {
          const f = circlePoly(z.anchor[1], z.anchor[0], 55);
          const over = Math.max(0, Math.min(1, ((shownValue(z) || 0) - S.thresholds.high) / 0.15));
          f.properties = { h: 70 + 170 * over };
          return f;
        }),
      });
    }
  }

  function syncSelection() {
    const z = S && S.zones.find((x) => x.id === selected);
    if (selMarker) { selMarker.remove(); selMarker = null; }
    if (z && !z.flagged) {
      const el = document.createElement('div');
      el.className = 'selmark';
      selMarker = new maplibregl.Marker({ element: el, anchor: 'center' }).setLngLat([z.anchor[1], z.anchor[0]]).addTo(map);
    }
    Object.keys(hotMarkers).forEach((id) => hotMarkers[id].getElement().classList.toggle('selected', id === selected));
  }

  function select(id, fly) {
    selected = id;
    if ($('#panel').classList.contains('hidden')) setPanel('open');
    syncSelection();
    renderDetail();
    if (fly) {
      const z = ZG.find((x) => x.id === id);
      if (z) map.easeTo({ center: [z.anchor[1], z.anchor[0]], zoom: Math.max(map.getZoom(), 14.2), duration: 900, padding: mapPadding() });
    }
  }

  // ====================================================================== render the state
  function reqDays() { return ahead || 7; }          // past and now use the standard 7-day forecast (what alerts use)
  function mlView() { return ahead > 0 && view === 'ml' && S && S.ml && S.ml.available; }
  function shownTier(z) { return mlView() && z.ml ? z.ml.tier : z.tier; }
  function shownValue(z) { return mlView() && z.ml ? z.ml.display : z.last_obs; }
  function applyView() {                    // what the map, markers and panel treat as 'flagged'
    S.zones.forEach((z) => { z.flagged = shownTier(z) === 'high' && !z.low_confidence; });
  }
  function latestReadingDate() {                     // the newest clear scene behind the map, not today's date
    const ages = S.zones.map((z) => z.obs_age_days).filter((a) => a != null);
    if (!ages.length) return S.as_of;
    return new Date(Date.parse(S.as_of) - Math.min.apply(null, ages) * DAY).toISOString().slice(0, 10);
  }
  function dayWord(n) { return n + ' day' + (n === 1 ? '' : 's'); }
  function renderAhead() {
    const avail = !!(S.ml && S.ml.available);
    $('#vMl').disabled = !avail;
    $('#vMl').title = avail ? 'The ML model trained for ' + dayWord(ahead) + ' ahead (' + (S.ml.name || '') + ')'
      : 'No ML model for ' + dayWord(ahead) + ' ahead yet: press "Train 1-7 day forecasts" in the Forecast Lab (L)';
    $('#vRule').setAttribute('aria-pressed', String(!mlView()));
    $('#vMl').setAttribute('aria-pressed', String(mlView()));
    const sk = (S.skill || []).find((x) => x.days === ahead);
    let t = '';
    if (!ahead) { $('#skillNote').textContent = ''; return; }
    if (sk) {
      t = 'Tested ' + dayWord(ahead) + ' ahead: latest reading scored ' + fmt2(sk.rule_f1) + ', ML (' + (sk.ml_name || '?') + ') ' + fmt2(sk.ml_f1) + ' (F1, higher is better). ' +
        (sk.use_ml ? 'Alerts use the ML for this day.' : 'Alerts use the latest reading.');
    } else if (!avail) t = 'The ML view needs the 1-7 day models: Forecast Lab (L) > Train 1-7 day forecasts.';
    if (avail) {
      const diff = S.zones.filter((z) => z.ml && z.rule_tier && (z.ml.tier === 'high') !== (z.rule_tier === 'high')).length;
      t += ' ML and the latest reading disagree on ' + diff + ' area' + (diff === 1 ? '' : 's') + '.';
    }
    $('#skillNote').textContent = t;
  }

  function render() {
    if (!S) return;
    applyView();
    // header chips and banners
    $('#modeChip').textContent = S.mode === 'replay' ? 'Replay of ' + S.as_of : 'Live';
    $('#modeChip').classList.toggle('warn', S.mode === 'replay');
    const banner = $('#dataBanner');
    const msgs = [];
    if (S.data_source === 'synthetic') msgs.push('SYNTHETIC DEMO DATA: not real');
    if (S.demo.enabled) msgs.push('Demo mode: the pin is a simulated location');
    if (!S.in_season) msgs.push('Outside the usual bloom season: alerts are paused');
    else if (S.late_season) msgs.push('Late season: a high reading is not a confirmed bloom');
    banner.hidden = msgs.length === 0;
    banner.textContent = msgs.join('. ');
    banner.classList.toggle('info', S.data_source !== 'synthetic');
    // toolbar
    $('#photonTxt').textContent = S.photon_connected ? 'Photon connected' : 'Photon off';
    $('#photonDot').classList.toggle('on', S.photon_connected);
    $('#demoToggle').checked = S.demo.enabled;
    $('#userSel').hidden = !S.demo.enabled;
    // timeline mode
    const replay = S.mode === 'replay';
    if (!replay && !playTimer && Date.now() - lastScrub > 1500) syncLiveSlider();
    $('#timeline').classList.toggle('future', !replay && ahead > 0);
    if (replay) { $('#dateLbl').textContent = S.as_of; $('#targetLbl').textContent = 'what the satellite showed by then'; }
    else if (ahead > 0) { $('#dateLbl').textContent = S.target_date; $('#targetLbl').textContent = '+' + dayWord(ahead) + ' after the latest reading (' + (mlView() ? 'ML model' : 'latest reading carried forward') + ')'; }
    else { $('#dateLbl').textContent = 'Now'; $('#targetLbl').textContent = 'latest satellite reading: ' + latestReadingDate(); }
    renderAhead();

    if (!selected || !S.zones.find((z) => z.id === selected)) {
      const f = S.zones.filter((z) => z.flagged).sort((a, b) => (shownValue(b) || 0) - (shownValue(a) || 0))[0] || S.zones[0];
      selected = f ? f.id : null;
    }
    syncHot(); syncSelection(); renderPin(); renderDetail(); renderReliab(); renderChatStatus(); renderAskState(); renderPhotoBadge();
    $('#foot').textContent = 'Satellite-based. Not a toxin test or a safety check. Official information: ' + S.official_info;
  }

  function renderReliab() {
    const r = S.reliability || {};
    const pct = (x) => (x == null ? 'n/a' : Math.round(x * 100) + '%');
    const how = S.method === 'persistence'
      ? 'Alerts use the latest clear satellite reading. A weather-based model was tested and did not catch more real events, so the simpler rule is used.'
      : 'Alerts blend the latest satellite reading with recent weather.';
    $('#reliab').innerHTML = '<p>' + esc(how) + '</p><p style="margin-top:8px">In past years (' + esc((r.test_years || []).join(', ')) + '), when an area was flagged, ' + pct(r.precision) +
      ' were still flagged a week later, and the method caught ' + pct(r.recall) + ' of all flagged readings. Those ' + esc(r.n_high == null ? '?' : r.n_high) +
      ' flagged readings fall on only ' + esc(r.n_high_dates == null ? '?' : r.n_high_dates) + ' dates, so treat these rates as rough.</p>';
  }

  function scaleHtml(z) {
    const lo = -0.1, hi = 0.4, pos = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
    const e = pos(S.thresholds.elevated), h = pos(S.thresholds.high);
    const dot = z.last_obs == null ? '' : '<span class="dot" style="left:' + pos(z.last_obs).toFixed(1) + '%"></span>';
    return '<div class="scale" style="--e:' + e.toFixed(1) + '%;--h:' + h.toFixed(1) + '%">' + dot + '</div>' +
      '<div class="scale-labels"><span style="left:0">lower</span><span style="left:' + h.toFixed(1) + '%;transform:translateX(-50%)">flag starts here</span><span style="right:0">higher</span></div>';
  }

  function trackLine(z) {
    const t = z.track;
    if (!t || !t.n || t.next_high == null) return '';
    if (z.tier === 'high') return 'In past years, about ' + Math.round(t.next_high * 10) + ' in 10 readings this high were still flagged a week later (from ' + t.n + ' past readings).';
    return 'In past years, about ' + Math.round(t.next_high * 100) + '% of readings like this reached the flag a week later.';
  }

  async function renderDetail() {
    const z = S.zones.find((x) => x.id === selected);
    const box = $('#detail');
    if (!z) { box.innerHTML = '<p class="muted">Select an area on the map.</p>'; return; }
    const age = z.obs_age_days == null ? '' : (z.obs_age_days <= 0 ? 'a satellite reading from today' : 'a satellite reading ' + z.obs_age_days + ' day' + (z.obs_age_days === 1 ? '' : 's') + ' old');
    const verif = z.verification ? '<p class="muted small" style="margin-top:6px">' + esc(dayWord(reqDays())) + ' later the satellite showed: <b>' + esc(z.verification.observed_tier === 'high' ? 'flagged' : z.verification.observed_tier) + '</b> (' + esc(z.verification.observed_date) + ').</p>' : '';
    const st = shownTier(z);
    const tw = (t) => (t === 'high' ? 'flagged' : t || 'n/a');
    const cmp = z.ml && ahead > 0 ? '<div class="compare"><div class="row2"><span>Latest reading carried forward</span><b>' + esc(tw(z.rule_tier)) + ' · ' + esc(z.last_obs == null ? '' : z.last_obs.toFixed(3)) + '</b></div>' +
      '<div class="row2"><span>ML model, ' + esc(dayWord(reqDays())) + ' ahead' + (S.ml && S.ml.name ? ' (' + esc(S.ml.name) + ')' : '') + '</span><b>' + esc(tw(z.ml.tier)) + ' · ' + esc(z.ml.value.toFixed(3)) + '</b></div>' +
      '<p class="muted small" style="margin-top:4px">' + (S.ml && S.ml.used_for_alerts ? 'Alerts use the ML for this day ahead (it clearly won in testing).' : 'Alerts use the latest reading; the ML is shown to compare.') + '</p></div>' : '';
    const low = z.low_confidence ? '<p class="lowconf">Low confidence: this stretch of river is only 30-60 m wide, so its satellite pixels mix in the banks. Shown for interest, never used for alerts.</p>' : '';
    box.innerHTML = '<h3>' + esc(z.name) + '</h3>' + (z.site_name ? '<p class="muted small">' + esc(z.site_name) + '</p>' : '') + '<span class="pill ' + esc(st) + '">' + esc(WORDS[st]) + '</span>' + low +
      '<p class="muted small" style="margin-top:8px">' + (age ? 'Based on ' + esc(age) + '.' : esc(z.reason || '')) + '</p>' + scaleHtml(z) + cmp +
      (reqDays() === 7 && !mlView() ? '<p class="muted small" style="margin-top:8px">' + esc(trackLine(z)) + '</p>' : '') + verif +
      '<h4>Satellite readings over time</h4><div id="chart"></div><h4>Photos from people nearby (checked by Grok)</h4><div id="photos" class="photos"></div>';
    const sel = selected;
    api('/api/history?zone=' + encodeURIComponent(sel)).then(({ data }) => { if (sel === selected && data.readings) drawChart(data); });
    api('/api/photos?zone=' + encodeURIComponent(sel)).then(({ data }) => { if (sel === selected && Array.isArray(data)) drawPhotos(data); });
  }

  function drawChart(h) {
    const W = 340, H = 132, pad = { l: 30, r: 8, t: 8, b: 20 };
    const all = h.readings.concat(h.after || []);
    if (!all.length) { $('#chart').innerHTML = '<p class="muted small">No readings yet.</p>'; return; }
    const t = (d) => new Date(d).getTime();
    const t0 = t(all[0].d), t1 = Math.max(t(all[all.length - 1].d), t(h.as_of));
    const vals = all.map((r) => r.v).concat([h.thresholds.high, -0.1]);
    const v0 = Math.min.apply(null, vals) - 0.02, v1 = Math.max.apply(null, vals) + 0.02;
    const x = (d) => pad.l + ((t(d) - t0) / Math.max(1, t1 - t0)) * (W - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - v0) / (v1 - v0)) * (H - pad.t - pad.b);
    const line = h.readings.map((r, i) => (i ? 'L' : 'M') + x(r.d).toFixed(1) + ',' + y(r.v).toFixed(1)).join(' ');
    const dots = h.readings.slice(-60).map((r) => '<circle cx="' + x(r.d).toFixed(1) + '" cy="' + y(r.v).toFixed(1) + '" r="2" fill="#10201b"/>').join('');
    const after = (h.after || []).map((r) => '<circle cx="' + x(r.d).toFixed(1) + '" cy="' + y(r.v).toFixed(1) + '" r="3.4" fill="none" stroke="#e0472d" stroke-width="1.6"/>').join('');
    const yh = y(h.thresholds.high).toFixed(1);
    $('#chart').innerHTML = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Satellite readings over time with the flag line">' +
      '<line x1="' + pad.l + '" x2="' + (W - pad.r) + '" y1="' + yh + '" y2="' + yh + '" stroke="#e0472d" stroke-dasharray="4 3"/>' +
      '<text x="' + (pad.l + 2) + '" y="' + (Number(yh) - 3) + '" font-size="10" fill="#e0472d">flag</text>' +
      '<path d="' + line + '" fill="none" stroke="#10201b" stroke-width="1.2" opacity=".5"/>' + dots + after +
      '<line x1="' + x(h.as_of).toFixed(1) + '" x2="' + x(h.as_of).toFixed(1) + '" y1="' + pad.t + '" y2="' + (H - pad.b) + '" stroke="#16a574" stroke-width="1.6"/>' +
      '<text x="' + pad.l + '" y="' + (H - 5) + '" font-size="10" fill="#5b6d66">' + esc(h.readings.length ? h.readings[0].d.slice(0, 7) : '') + '</text>' +
      '<text x="' + (W - pad.r) + '" y="' + (H - 5) + '" text-anchor="end" font-size="10" fill="#5b6d66">' + esc(h.as_of) + (h.after ? ' (circles: what came next)' : '') + '</text></svg>';
  }

  function drawPhotos(list) {
    const box = $('#photos');
    if (!box) return;
    if (!list.length) { box.innerHTML = '<p class="muted small" style="grid-column:1/-1">No verified photos yet. People can text a photo to the bot.</p>'; return; }
    box.innerHTML = list.map((p) => '<img src="/photos/' + p.id + '.jpg" alt="Community photo: ' + esc((BLOOM_WORDS[p.bloom] || 'checked by Grok').toLowerCase()) + '" title="' + esc((BLOOM_WORDS[p.bloom] || '') + (p.description ? ': ' + p.description : '')) + '" loading="lazy"' + (p.bloom === 'bloom' ? ' class="bloom"' : '') + '>').join('');
    box.querySelectorAll('img').forEach((im) => im.addEventListener('click', () => { $('#lightbox img').src = im.src; $('#lightbox').hidden = false; }));
  }
  $('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });

  // ====================================================================== panels
  function setPanel(mode) {                       // 'open' | 'min' | 'hidden'
    const p = $('#panel');
    p.classList.toggle('hidden', mode === 'hidden');
    p.classList.toggle('min', mode === 'min');
    $('#btnPanel').setAttribute('aria-pressed', String(mode !== 'hidden'));
    prefs.set('panel', mode);
  }
  $('#panelMin').addEventListener('click', () => setPanel($('#panel').classList.contains('min') ? 'open' : 'min'));
  $('#panelHide').addEventListener('click', () => setPanel('hidden'));
  $('#btnPanel').addEventListener('click', () => setPanel($('#panel').classList.contains('hidden') ? 'open' : 'hidden'));
  $('#btn3d').addEventListener('click', () => {
    const on = map.getPitch() < 10;
    map.easeTo({ pitch: on ? 55 : 0, bearing: on ? -18 : 0, duration: 700 });
    prefs.set('pitch', on ? 52 : 0);
    $('#btn3d').setAttribute('aria-pressed', String(on));
  });

  // ====================================================================== alerts feed
  let feedAll = [], feedShowAll = false;
  async function refreshFeed() {
    const { ok, data } = await api('/api/feed');
    if (!ok || !Array.isArray(data)) return;
    feedAll = data;
    renderFeed();
  }
  function renderFeed() {
    const cleared = prefs.get('clearedAt', '');
    const items = feedAll.filter((f) => !cleared || f.ts > cleared);
    $('#alertCount').textContent = String(items.length);
    const shown = feedShowAll ? items : items.slice(0, 3);
    const ul = $('#feed');
    ul.innerHTML = '';
    if (!shown.length) {
      ul.innerHTML = '<li class="empty">' + (feedAll.length ? 'Cleared. New alerts will show here.' : 'Nothing sent yet. Drag the pin into a flagged area.') + '</li>';
    }
    shown.forEach((f) => {
      const li = document.createElement('li');
      const time = new Date(f.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const where = f.zone + (f.distance_m != null ? ' (' + f.distance_m + ' m)' : '');
      li.innerHTML = '<div class="top"><span><b>' + esc(f.to) + '</b> near ' + esc(where) + '</span><span class="muted">' + esc(time) + '</span></div><div class="tags"></div><pre></pre>';
      li.querySelector('pre').textContent = f.text;
      const tags = li.querySelector('.tags');
      const add = (txt, cls) => { const s = document.createElement('span'); s.className = 'tag ' + (cls || ''); s.textContent = txt; tags.appendChild(s); };
      if (f.simulated) add('simulated');
      if (f.delivered) add('sent', 'ok'); else add('not sent' + (f.note ? ': ' + f.note.slice(0, 54) : ''), 'warn');
      li.addEventListener('click', () => li.classList.toggle('open'));
      ul.appendChild(li);
    });
    const more = $('#alertsMore');
    const hidden = items.length - shown.length;
    more.hidden = !(hidden > 0 || (feedShowAll && items.length > 3));
    more.textContent = feedShowAll ? 'Show fewer' : 'Show ' + hidden + ' more';
  }
  $('#alertsMore').addEventListener('click', () => { feedShowAll = !feedShowAll; renderFeed(); });
  $('#alertsClear').addEventListener('click', () => { prefs.set('clearedAt', feedAll.length ? feedAll[0].ts : new Date().toISOString()); renderFeed(); });
  $('#alertsToggle').addEventListener('click', () => {
    const el = $('#alerts');
    const min = !el.classList.contains('min');
    el.classList.toggle('min', min);
    $('#alertsToggle').setAttribute('aria-expanded', String(!min));
    prefs.set('alertsMin', min);
  });

  // ====================================================================== timeline: past ... Now ... +7 days
  // One slider. Left of Now: replay what the satellite showed on that date. Now: the latest reading (live).
  // Right of Now (current season only): 1-7 days ahead of the latest reading.
  let asOfSeq = 0, scrubTimer = null, playTimer = null, lastScrub = 0;
  function yearStart(y) { return Date.UTC(y, 4, 1); }
  function todayStr() { return (FR && FR.today) || new Date().toISOString().slice(0, 10); }
  function curYear() { return Number(todayStr().slice(0, 4)); }
  function nowIdx(y) { return Number(y) === curYear() ? Math.max(0, Math.round((Date.parse(todayStr()) - yearStart(Number(y))) / DAY)) : null; }
  function sliderMax(y) {
    const n = nowIdx(y);
    if (n !== null) return n + 7;
    let max = 183;
    if (FR && String(y) === String(FR.max).slice(0, 4)) max = Math.min(183, Math.max(0, Math.round((Date.parse(FR.max) + 14 * DAY - yearStart(y)) / DAY)));
    return max;
  }
  function idxDate(i) { return new Date(yearStart(Number($('#yearSel').value)) + i * DAY).toISOString().slice(0, 10); }
  function sliderDate() { return idxDate(Number($('#slider').value)); }
  function layoutSlider() {
    const y = $('#yearSel').value, n = nowIdx(y), max = sliderMax(y);
    $('#slider').max = String(max);
    $('#tlEnd').textContent = n !== null ? '+7 days' : 'Oct';
    const tick = $('#nowTick');
    tick.hidden = n === null;
    if (n !== null) {
      const pct = max ? n / max : 0;
      tick.style.left = 'calc(' + (pct * 100).toFixed(2) + '% + ' + (8 - 16 * pct).toFixed(1) + 'px)';    // follow the thumb's real travel
      const past = (pct * 100).toFixed(2);
      $('#slider').style.background = 'linear-gradient(90deg, transparent ' + past + '%, rgba(42,120,214,.16) ' + past + '%)';
    } else $('#slider').style.background = '';
  }
  function syncLiveSlider() {                       // keep the thumb on Now (+ahead) while live
    const y = curYear();
    if ($('#yearSel').value !== String(y) && [...$('#yearSel').options].some((o) => o.value === String(y))) $('#yearSel').value = String(y);
    layoutSlider();
    const n = nowIdx($('#yearSel').value);
    if (n !== null) $('#slider').value = String(n + ahead);
  }
  function setSliderTo(dateStr) {
    const d = Date.parse(dateStr), y = new Date(d).getUTCFullYear();
    if ($('#yearSel').value !== String(y)) $('#yearSel').value = String(y);
    layoutSlider();
    $('#slider').value = String(Math.max(0, Math.min(Number($('#slider').max), Math.round((d - yearStart(y)) / DAY))));
    $('#dateLbl').textContent = dateStr;
  }
  async function postAsOf(dateStr, notify) {
    const seq = ++asOfSeq;
    const r = await api('/api/as_of', { as_of: dateStr, notify: notify });
    if (seq !== asOfSeq) return;                  // a newer request is already on its way
    await refresh();
    if (notify) refreshFeed();
    if (r.ok && r.data.alerts) toast(r.data.alerts + ' alert' + (r.data.alerts === 1 ? '' : 's') + ' sent for this date.');
  }
  // move to slider position i: past (replay), now, or days ahead
  function goIndex(i, notify) {
    const n = nowIdx($('#yearSel').value);
    if (n !== null && i >= n) {
      const wasLive = S && S.mode === 'live';
      ahead = Math.min(7, i - n);
      heatUrl = null;
      $('#dateLbl').textContent = ahead ? '+' + dayWord(ahead) : 'Now';
      return wasLive ? refresh() : postAsOf(null, notify);
    }
    ahead = 0;
    $('#dateLbl').textContent = idxDate(i);
    return postAsOf(idxDate(i), notify);
  }
  function goReplay(dateStr, notify) { stopPlay(); ahead = 0; setSliderTo(dateStr); return postAsOf(dateStr, notify !== false); }
  function goNow() { stopPlay(); ahead = 0; syncLiveSlider(); goIndex(Number($('#slider').value), true); }
  $('#nowBtn').addEventListener('click', goNow);
  $('#jump').addEventListener('click', () => goReplay(FR.suggested));
  $('#yearSel').addEventListener('change', () => { stopPlay(); layoutSlider(); const n = nowIdx($('#yearSel').value); $('#slider').value = String(n !== null ? n : 120); goIndex(Number($('#slider').value), true); });
  $('#slider').addEventListener('input', () => {
    stopPlay();
    lastScrub = Date.now();
    const i = Number($('#slider').value), n = nowIdx($('#yearSel').value);
    $('#dateLbl').textContent = n !== null && i >= n ? (i === n ? 'Now' : '+' + dayWord(Math.min(7, i - n))) : sliderDate();
    clearTimeout(scrubTimer);
    scrubTimer = setTimeout(() => goIndex(i, false), 250);   // scrubbing never texts anyone
  });
  $('#slider').addEventListener('change', () => { clearTimeout(scrubTimer); goIndex(Number($('#slider').value), true); });
  function stepWeek(dir) { stopPlay(); const i = Math.max(0, Math.min(Number($('#slider').max), Number($('#slider').value) + 7 * dir)); $('#slider').value = String(i); goIndex(i, true); }
  $('#stepBack').addEventListener('click', () => stepWeek(-1));
  $('#stepFwd').addEventListener('click', () => stepWeek(1));
  function stopPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    $('#playIcon').innerHTML = '<path d="M7 5l9 5-9 5z" fill="currentColor"/>';
    $('#playBtn').setAttribute('aria-label', 'Play');
  }
  $('#playBtn').addEventListener('click', () => {
    if (playTimer) { stopPlay(); return; }
    $('#playIcon').innerHTML = '<path d="M6 5h3v10H6zM11 5h3v10h-3z" fill="currentColor"/>';
    $('#playBtn').setAttribute('aria-label', 'Pause');
    const n = nowIdx($('#yearSel').value), end = n !== null ? n : Number($('#slider').max);
    if (Number($('#slider').value) >= end) $('#slider').value = '0';
    playTimer = setInterval(() => {
      const i = Math.min(end, Number($('#slider').value) + 2);
      $('#slider').value = String(i);
      if (i >= end) { stopPlay(); goIndex(i, false); return; }   // stops at Now
      $('#dateLbl').textContent = sliderDate();
      postAsOf(sliderDate(), false);
    }, 700);
  });
  // days ahead only: colour the map by the latest reading (the simple rule) or by the ML model for that day
  function setView(v) { view = v; prefs.set('view', v); heatUrl = null; render(); updateHeat(); }
  $('#vRule').addEventListener('click', () => setView('rule'));
  $('#vMl').addEventListener('click', () => setView('ml'));

  // ====================================================================== demo mode and the pin
  function syncPinRing() {
    const src = map.getSource('pin-ring');
    if (!src) return;
    if (!pinMarker || !S || !S.demo.enabled) { src.setData({ type: 'FeatureCollection', features: [] }); return; }   // no pin, no ring
    const ll = pinMarker.getLngLat();
    src.setData({ type: 'FeatureCollection', features: [circlePoly(ll.lng, ll.lat, S.demo.radius_m)] });
  }
  function startPoint() {                         // about 1 km north of a flagged area, so the first drag shows the alert
    const f = S.zones.find((z) => z.flagged) || S.zones[0];
    return [f.anchor[1], f.anchor[0] + 0.009];
  }
  function renderPin() {
    if (!S.demo.enabled) {
      if (pinMarker) { pinMarker.remove(); pinMarker = null; }
      syncPinRing();
      return;
    }
    const where = S.demo.pin ? [S.demo.pin.lon, S.demo.pin.lat] : startPoint();
    if (!pinMarker) {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'width:0;height:0';
      wrap.innerHTML = '<div class="pinicon"></div>';
      pinMarker = new maplibregl.Marker({ element: wrap, draggable: true, anchor: 'center' }).setLngLat(where).addTo(map);
      pinMarker.on('dragstart', () => { pinDragging = true; });
      pinMarker.on('drag', syncPinRing);
      pinMarker.on('dragend', () => { pinDragging = false; syncPinRing(); sendPin(pinMarker.getLngLat()); });
    } else if (S.demo.pin && !pinDragging) {
      pinMarker.setLngLat(where);
    }
    syncPinRing();
  }
  function alertBasis() {           // the pin checks what alerts use, which may differ from the map's +days / ML view
    return ahead > 0 || mlView() ? ' Alerts use the latest reading (7-day forecast), not the +' + ahead + ' day ' + (mlView() ? 'ML ' : '') + 'view on the map.' : '';
  }
  async function sendPin(ll) {
    const r = await api('/api/pin', { lat: ll.lat, lon: ll.lng });
    if (!r.ok) { toast(r.data.error || 'Could not move the pin.'); return; }
    const d = r.data, n = d.nearest;
    if (d.alerted) toast('Alert sent to the selected phone: near ' + n.name + ' (~' + n.distance_m + ' m).');
    else if (n.within_radius && n.flagged) toast(((d.result.skipped || [])[0] || {}).reason ? 'Not sent: ' + d.result.skipped[0].reason : 'Already alerted a moment ago.');
    else if (n.within_radius && n.low_confidence && n.tier === 'high') toast('Near ' + n.name + ': high reading, but this narrow stretch is low-confidence, so it never sends alerts.');
    else if (n.within_radius) toast('Near ' + n.name + ': ' + (n.tier === 'unknown' ? 'no recent reading' : 'reading is ' + (n.tier === 'elevated' ? 'raised but below the flag' : 'low')) + ', so no alert.' + alertBasis());
    else toast('No flagged area within ' + S.demo.radius_m + ' m (nearest: ' + n.name + ', ' + n.distance_m + ' m).' + alertBasis());
    refreshFeed(); refresh();
  }
  async function loadUsers() {
    const { data } = await api('/api/users');
    const sel = $('#userSel');
    if (!Array.isArray(data)) return;
    const cur = (S && S.demo.user_id) || sel.value;
    sel.innerHTML = data.length ? data.map((u) => '<option value="' + esc(u.user_id) + '">' + esc(u.label) + (u.reachable ? '' : ' (cannot reach)') + '</option>').join('') : '<option value="">No phones yet</option>';
    if (cur) sel.value = cur;
  }
  $('#demoToggle').addEventListener('change', async (e) => {
    await loadUsers();
    const r = await api('/api/demo', { enabled: e.target.checked, user_id: $('#userSel').value || null });
    if (!r.ok) { e.target.checked = false; toast(r.data.error || 'Could not start demo mode.'); }
    await refresh();
  });
  $('#userSel').addEventListener('change', async () => { await api('/api/demo', { enabled: true, user_id: $('#userSel').value }); refresh(); });

  // ====================================================================== chat with voice
  const chat = { history: [], speak: prefs.get('speak', true), busy: false, rec: null, audio: null };
  const SUGGESTIONS = ['Which areas are flagged?', 'Why is it flagged?', 'Can you predict next week?', 'Is it safe to swim?'];

  function renderChatStatus() {
    const c = (S && S.capabilities) || {};
    $('#chatStatus').textContent = c.grok ? 'Grok ' + c.model + ' with voice ' + c.voice : 'Grok is not connected: answering from fixed wording';
  }
  function addMsg(role, text, extra) {
    const el = document.createElement('div');
    el.className = 'msg ' + role + (extra && extra.error ? ' error' : '') + (extra && extra.typing ? ' typing' : '');
    el.textContent = text;
    if (extra && extra.meta) { const m = document.createElement('span'); m.className = 'meta'; m.textContent = extra.meta; el.appendChild(m); }
    $('#chatLog').appendChild(el);
    $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
    return el;
  }
  function openChat(open) {
    $('#chat').hidden = !open;
    if (open) {
      if (!$('#chatLog').children.length) addMsg('bot', 'Hi. I can tell you what the satellite data shows for the Charles River and the nearby lakes: which areas are flagged, why, and how reliable it is. I cannot tell you whether the water is safe.');
      $('#chatInput').focus();
    } else { stopSpeaking(); }
  }
  $('#chatFab').addEventListener('click', () => openChat($('#chat').hidden));
  $('#chatClose').addEventListener('click', () => openChat(false));
  $('#speakToggle').setAttribute('aria-pressed', String(chat.speak));
  $('#speakToggle').addEventListener('click', () => { chat.speak = !chat.speak; prefs.set('speak', chat.speak); $('#speakToggle').setAttribute('aria-pressed', String(chat.speak)); if (!chat.speak) stopSpeaking(); });
  $('#chatSuggest').innerHTML = SUGGESTIONS.map((s) => '<button type="button">' + esc(s) + '</button>').join('');
  $('#chatSuggest').addEventListener('click', (e) => { if (e.target.tagName === 'BUTTON') submitChat(e.target.textContent); });
  $('#chatForm').addEventListener('submit', (e) => { e.preventDefault(); const t = $('#chatInput').value.trim(); if (t) { $('#chatInput').value = ''; submitChat(t); } });

  async function submitChat(text) {
    if (chat.busy) return;
    chat.busy = true;
    stopSpeaking();
    addMsg('user', text);
    const typing = addMsg('bot', 'Thinking...', { typing: true });
    const r = await api('/api/chat', { message: text, history: chat.history.slice(-6) });
    typing.remove();
    chat.busy = false;
    if (!r.ok) { addMsg('bot', r.data.error || 'Something went wrong. Please try again.', { error: true }); return; }
    const d = r.data;
    chat.history.push({ role: 'user', content: text }, { role: 'assistant', content: d.reply });
    const meta = d.source === 'grok' ? 'Grok' + (d.replaced ? ' (answer replaced for safety)' : '') : d.source === 'rule' ? 'Fixed safety answer' : 'Fixed wording (Grok not connected)';
    addMsg('bot', d.reply, { meta: meta });
    if (chat.speak) speak(d.reply);
  }

  function stopSpeaking() {
    if (chat.audio) { try { chat.audio.pause(); } catch (e) { /* ignore */ } chat.audio = null; }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  }
  function browserSpeak(text) {
    if (!('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }
  async function speak(text) {
    const caps = (S && S.capabilities) || {};
    if (caps.grok) {
      try {
        const r = await fetch('/api/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) });
        if (r.ok && (r.headers.get('content-type') || '').indexOf('audio') === 0) {
          const url = URL.createObjectURL(await r.blob());
          chat.audio = new Audio(url);
          chat.audio.onended = () => URL.revokeObjectURL(url);
          await chat.audio.play();
          return;
        }
      } catch (e) { /* fall through to the browser voice */ }
    }
    browserSpeak(text);
  }

  // ---- microphone: record, convert to a small WAV, let Grok transcribe (browser recognition as the fallback)
  function encodeWav(samples, rate) {
    const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
    const wr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    wr(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); wr(8, 'WAVE'); wr(12, 'fmt '); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    wr(36, 'data'); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    return new Blob([buf], { type: 'audio/wav' });
  }
  async function toWav16k(blob) {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const audio = await ac.decodeAudioData(await blob.arrayBuffer());
    if (ac.close) ac.close();
    const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(audio.duration * 16000)), 16000);
    const src = off.createBufferSource();
    src.buffer = audio; src.connect(off.destination); src.start();
    const rendered = await off.startRendering();
    return encodeWav(rendered.getChannelData(0), 16000);
  }
  function setMic(on) { $('#micBtn').classList.toggle('rec', on); $('#micBtn').setAttribute('aria-label', on ? 'Stop recording' : 'Speak your question'); }
  function browserListen() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { toast('Voice input needs Chrome or Edge, or a Grok key in .env.'); return; }
    const r = new SR();
    r.lang = 'en-US'; r.interimResults = false;
    r.onresult = (e) => submitChat(e.results[0][0].transcript);
    r.onerror = () => toast('I could not hear that. Try again or type your question.');
    r.onend = () => setMic(false);
    setMic(true); r.start();
  }
  async function toggleMic() {
    if (chat.rec) { chat.rec.stop(); return; }
    const caps = (S && S.capabilities) || {};
    if (!caps.grok || !navigator.mediaDevices || !window.MediaRecorder) { browserListen(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch (e) { toast('Microphone access was blocked. You can still type.'); return; }
    const chunks = [];
    const mr = new MediaRecorder(stream);
    chat.rec = mr;
    mr.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      chat.rec = null; setMic(false);
      try {
        const wav = await toWav16k(new Blob(chunks, { type: mr.mimeType }));
        const r = await fetch('/api/stt', { method: 'POST', headers: { 'X-Audio-Type': 'audio/wav' }, body: wav });
        const d = await r.json();
        if (!r.ok) { toast(d.error || 'Could not transcribe that.'); return; }
        if (d.text) submitChat(d.text); else toast('I did not catch anything. Try again.');
      } catch (e) { toast('Could not process the recording. You can type instead.'); }
    };
    mr.start(); setMic(true);
    setTimeout(() => { if (chat.rec === mr && mr.state === 'recording') mr.stop(); }, 20000);
  }
  $('#micBtn').addEventListener('click', toggleMic);

  // ====================================================================== forecast lab and ask the data
  const METHOD_NAMES = { persistence: 'Latest reading carried forward', climatology: 'Seasonal average', hgb: 'Weather model (plain)', hgb_tuned: 'Weather model (tuned alerts)',
    logit: 'Logistic model (tuned alerts)', delta: 'Predict the change, not the level', classifier: "Predict 'high' directly", no_weather: 'Model without weather',
    neighbors: 'Model + neighbouring zones', memory: 'Model + longer memory', importance: 'Which inputs matter' };
  let labPoll = 0;
  function openLab(open) {
    $('#labModal').hidden = !open;
    if (open) { loadLatestLab(); renderAskState(); }
  }
  $('#btnLab').addEventListener('click', () => openLab(true));
  $('#labClose').addEventListener('click', () => openLab(false));
  $('#labModal').addEventListener('click', (e) => { if (e.target.id === 'labModal') openLab(false); });

  function runnerChip(label) {
    const chip = $('#labRunner');
    chip.textContent = label || '-';
    chip.classList.toggle('warn', !!label && label.toLowerCase().indexOf('daytona') === -1);
  }
  async function loadLatestLab() {
    const { data } = await api('/api/lab/latest');
    if (data && data.summary) renderLab(data); else { $('#labResult').innerHTML = '<p class="muted small" style="margin-top:14px">No results yet. Run the quick lab (about a minute) to compare methods.</p>'; runnerChip(((S || {}).capabilities || {}).daytona ? 'Daytona ready' : 'Daytona not connected'); }
  }
  function renderLab(p) {
    runnerChip(p.runner);
    $('#labStamp').textContent = p.finished_at ? 'Last run ' + p.finished_at : '';
    const b = $('#labBanner');
    b.hidden = p.data_source !== 'synthetic';
    b.textContent = 'SYNTHETIC DEMO DATA: these results describe made-up data, not the real river.';
    const s = p.summary;
    let html = '<div class="headline">' + esc(s.headline) + '</div>';
    if (s.trend) html += '<p class="muted small" style="margin-top:8px">' + esc(s.trend) + '</p>';
    (p.notes || []).forEach((n) => { html += '<p class="small" style="margin-top:8px;color:var(--alert-ink)">' + esc(n) + '</p>'; });
    s.groups.forEach((g) => {
      const win = g.verdict.indexOf('clearly better') === 0;
      html += '<div class="group"><h4>' + g.horizon + ' days ahead, flag the top ' + Math.round((1 - g.high_q) * 100) + '%<span class="verdict ' + (win ? 'win' : '') + '">' + esc(g.verdict) + '</span></h4>';
      g.rows.forEach((r) => {
        const f1 = r.f1 || 0, lo = (r.f1_range || [f1, f1])[0], hi = (r.f1_range || [f1, f1])[1];
        html += '<div class="row"><span>' + esc(METHOD_NAMES[r.method] || r.method) + '</span><div class="barwrap" title="score ' + f1.toFixed(2) + ', range ' + lo.toFixed(2) + ' to ' + hi.toFixed(2) + '">' +
          '<div class="fill ' + (r.method === 'persistence' ? 'base' : '') + '" style="width:' + (f1 * 100).toFixed(1) + '%"></div><div class="range" style="left:' + (lo * 100).toFixed(1) + '%;width:' + Math.max(0.5, (hi - lo) * 100).toFixed(1) + '%"></div></div>' +
          '<span class="nums">score ' + f1.toFixed(2) + ' (' + lo.toFixed(2) + ' to ' + hi.toFixed(2) + ') &middot; caught ' + Math.round((r.recall || 0) * 100) + '%</span></div>';
      });
      html += '</div>';
    });
    if ((s.confirmations || []).length) {
      html += '<div class="group"><h4>Check on the held-out ' + esc(s.confirmations[0].year) + ' season</h4>';
      s.confirmations.forEach((c) => {
        html += '<p class="small" style="padding:4px 0;border-top:1px solid var(--line)"><b>' + c.horizon + ' days ahead:</b> ' + esc(c.name) + ' scored ' + (c.f1 || 0).toFixed(2) + ' vs ' + (c.base_f1 || 0).toFixed(2) +
          ' for the simple rule: <span class="verdict ' + (c.clear ? 'win' : '') + '">' + esc(c.verdict) + '</span></p>';
      });
      html += '</div>';
    }
    if (s.importance && s.importance.importance) {
      html += '<p class="small" style="margin-top:12px">Inputs the model leans on most: ' + s.importance.importance.slice(0, 4).map((r) => esc(r.label)).join(', ') + '. Open <b>Science</b> (S) for the full chart.</p>';
    }
    html += '<p class="muted small" style="margin-top:14px">' + esc(s.caution || '') + ' The bar is the score (higher is better); the dark line is the 95% range.' + (p.holdout ? ' Screening used seasons before ' + esc(p.holdout) + '.' : '') + '</p>';
    $('#labResult').innerHTML = html;
  }
  async function pollJob(kind, id, onTick) {
    const mine = ++labPoll;
    for (let i = 0; i < 400 && mine === labPoll; i++) {
      const { ok, data } = await api('/api/' + kind + '/status?id=' + encodeURIComponent(id));
      if (!ok) return { status: 'error', error: (data && data.error) || 'Lost track of the job.' };
      onTick(data);
      if (data.status !== 'running') return data;
      await sleep(1200);
    }
    return { status: 'error', error: 'Gave up waiting.' };
  }
  async function runLab(mode) {
    const r = await api('/api/lab/run', { mode: mode });
    if (!r.ok) { toast(r.data.error || 'Could not start the lab.'); return; }
    $('#labQuick').disabled = $('#labFull').disabled = $('#labTrain').disabled = true;
    $('#labProgress').hidden = false; $('#labLog').hidden = false;
    $('#labBar').style.width = '4%';
    const done = await pollJob('lab', r.data.job_id, (j) => {
      runnerChip(j.runner || 'starting');
      $('#labBar').style.width = Math.max(4, j.total ? (j.done / j.total) * 100 : 4) + '%';
      $('#labLog').textContent = (j.log || []).join('\n');
    });
    $('#labQuick').disabled = $('#labFull').disabled = $('#labTrain').disabled = false;
    $('#labProgress').hidden = true; $('#labLog').hidden = true;
    if (done.status === 'done' && done.result) renderLab(done.result);
    else $('#labResult').innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:14px">The lab stopped: ' + esc(done.error || 'unknown problem') + '</p>';
  }
  async function runTrain() {
    const r = await api('/api/train/run', {});
    if (!r.ok) { toast(r.data.error || 'Could not start the training.'); return; }
    $('#labQuick').disabled = $('#labFull').disabled = $('#labTrain').disabled = true;
    $('#labProgress').hidden = false; $('#labLog').hidden = false;
    $('#labBar').style.width = '2%';
    const done = await pollJob('train', r.data.job_id, (j) => {
      runnerChip(j.runner || 'starting');
      $('#labBar').style.width = Math.max(2, j.total ? (j.done / j.total) * 100 : 2) + '%';
      $('#labLog').textContent = (j.log || []).join('\n');
    });
    $('#labQuick').disabled = $('#labFull').disabled = $('#labTrain').disabled = false;
    $('#labProgress').hidden = true; $('#labLog').hidden = true;
    if (done.status === 'done' && done.result) { $('#labResult').innerHTML = trainHtml(done.result); refresh(); toast('1-7 day models trained: try the ML model toggle on the timeline.'); }
    else $('#labResult').innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:14px">The training stopped: ' + esc(done.error || 'unknown problem') + '</p>';
  }
  function trainHtml(p) {
    const s = p.summary || {};
    let h = '<div class="headline">' + esc(s.headline || '') + '</div><p class="muted small" style="margin-top:6px">' + esc(p.n) + ' experiments on ' + esc(p.runner) + ', ' + esc(p.finished_at || '') +
      '. Each day ahead: 5 model ideas x 3 settings, screened on seasons before ' + esc(p.holdout) + '; the best is then checked on ' + esc(p.holdout) + ', which it never saw.</p>';
    (p.notes || []).forEach((n) => { h += '<p class="small" style="margin-top:6px;color:var(--alert-ink)">' + esc(n) + '</p>'; });
    h += '<table class="hdays"><tr><th>Days ahead</th><th>Latest reading</th><th>Best ML model</th><th>On ' + esc(p.holdout) + '</th><th>Alerts use</th></tr>' +
      (s.days || []).map((d) => '<tr><td class="num">+' + d.days + '</td><td class="num">' + fmt2(d.rule_f1) + '</td><td>' + fmt2(d.ml_f1) + ' <span class="muted">' + esc(d.ml_name || '') + ', ' + esc(d.ml_setting_name || '') + '</span></td>' +
        '<td class="num">ML ' + fmt2(d.holdout_ml_f1) + ' / rule ' + fmt2(d.holdout_rule_f1) + '</td><td>' + (d.use_ml ? '<b>ML</b>' : 'latest reading') + '</td></tr>').join('') + '</table>' +
      '<p class="muted small" style="margin-top:6px">Scores are F1 at catching readings in the top 10% (higher is better). The ML only drives alerts where it was clearly better on earlier seasons and also on the unseen one.</p>';
    return h;
  }
  $('#labTrain').addEventListener('click', runTrain);
  $('#labQuick').addEventListener('click', () => runLab('quick'));
  $('#labFull').addEventListener('click', () => runLab('full'));

  function renderAskState() {
    const c = (S && S.capabilities) || {};
    const ready = !!(c.daytona && c.grok);
    $('#askBtn').disabled = !ready;
    $('#askHint').textContent = ready ? '' : 'Needs DAYTONA_API_KEY and XAI_API_KEY in .env (and: pip install daytona). Until then this stays off, because AI-written code is never run on this computer.';
  }
  $('#askForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('#askInput').value.trim();
    if (!q) return;
    const out = $('#askResult');
    out.innerHTML = '<p class="muted small" style="margin-top:10px">Asking Grok to write the analysis...</p>';
    const r = await api('/api/ask', { question: q });
    if (!r.ok) { out.innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:10px">' + esc(r.data.error || 'Could not start.') + '</p>'; return; }
    $('#askBtn').disabled = true;
    const done = await pollJob('ask', r.data.job_id, (j) => { out.innerHTML = '<p class="muted small" style="margin-top:10px">' + esc((j.log || []).slice(-1)[0] || 'working...') + '</p>'; });
    renderAskState();
    if (done.status !== 'done' || !done.result) { out.innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:10px">' + esc(done.error || 'Something went wrong.') + '</p>'; return; }
    const a = done.result;
    let html = '';
    if (a.refused) html += '<p class="small" style="color:var(--alert-ink);margin-top:10px">' + esc(a.refused) + '</p>';
    else html += '<p class="small muted" style="margin-top:10px">Ran in <b>' + esc(a.runner) + '</b> in ' + esc(a.seconds) + ' s.</p>';
    if (a.output != null) html += '<div class="outbox">' + esc(a.output || '(no output)') + '</div>';
    html += '<details class="codebox"' + (a.refused ? ' open' : '') + '><summary>The code Grok wrote</summary><pre>' + esc(a.code) + '</pre></details>';
    out.innerHTML = html;
  });

  // ====================================================================== small chart kit (plain SVG, one shared tooltip)
  const SERIES = ['#2a78d6', '#eb6834', '#1baf7a'];          // fixed order: never cycled
  const fmtPct = (x) => (x == null ? 'n/a' : Math.round(x * 100) + '%');
  const fmt2 = (x) => (x == null ? 'n/a' : Number(x).toFixed(2));
  const fmtUsd = (x) => (x == null ? 'n/a' : x >= 1e6 ? '$' + (x / 1e6).toFixed(x >= 1e7 ? 0 : 1) + 'M' : x >= 1e3 ? '$' + Math.round(x / 1e3) + 'k' : '$' + Math.round(x));
  function showTip(ev, title, rows) {                       // rows: [{name, value, color}]; text only, never HTML
    const tip = $('#vizTip');
    tip.textContent = '';
    const t = document.createElement('div'); t.className = 't'; t.textContent = title; tip.appendChild(t);
    rows.forEach((r) => {
      const row = document.createElement('div'); row.className = 'r';
      const n = document.createElement('span'); n.className = 'n';
      if (r.color) { const k = document.createElement('i'); k.style.background = r.color; n.appendChild(k); }
      n.appendChild(document.createTextNode(r.name));
      const b = document.createElement('b'); b.textContent = r.value;
      row.appendChild(n); row.appendChild(b); tip.appendChild(row);
    });
    tip.hidden = false;
    const x = Math.min(window.innerWidth - tip.offsetWidth - 12, ev.clientX + 14);
    const y = Math.min(window.innerHeight - tip.offsetHeight - 12, ev.clientY + 14);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { $('#vizTip').hidden = true; }
  function legendHtml(items, dot) {
    return '<div class="legend-row">' + items.map((it) => '<span><span class="key' + (dot ? ' dot' : '') + '" style="background:' + it.color + '"></span>' + esc(it.name) + '</span>').join('') + '</div>';
  }
  function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v || 1))); return Math.ceil(v / p) * p; }

  // line chart over categories (years); every series at an X in one tooltip
  function lineChart(el, cats, series, opt) {
    const W = 460, H = 210, pad = { l: 44, r: 92, t: 10, b: 24 };
    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    const y1 = niceMax(Math.max.apply(null, all) * 1.08), y0 = 0;
    const x = (i) => pad.l + (cats.length === 1 ? 0.5 : i / (cats.length - 1)) * (W - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);
    let g = '';
    for (let k = 0; k <= 4; k++) {
      const v = y0 + (k / 4) * (y1 - y0);
      g += '<line x1="' + pad.l + '" x2="' + (W - pad.r) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="var(--grid)"/><text x="' + (pad.l - 6) + '" y="' + (y(v) + 3.5) + '" text-anchor="end">' + v.toFixed(opt.decimals == null ? 3 : opt.decimals) + '</text>';
    }
    cats.forEach((c, i) => { g += '<text x="' + x(i) + '" y="' + (H - 6) + '" text-anchor="middle">' + esc(c) + '</text>'; });
    series.forEach((s, si) => {
      const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
      g += '<path d="' + pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('') + '" fill="none" stroke="' + SERIES[si] + '" stroke-width="2" stroke-linejoin="round"/>';
      pts.forEach((p) => { g += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="4" fill="' + SERIES[si] + '" stroke="#fff" stroke-width="2"/>'; });
      const last = pts[pts.length - 1];
      if (last) g += '<text class="lbl" x="' + (last[0] + 8) + '" y="' + (last[1] + 4 + (si - 1) * 3) + '">' + esc(s.short || s.name) + '</text>';   // direct label
    });
    g += '<line class="xh" x1="0" x2="0" y1="' + pad.t + '" y2="' + (H - pad.b) + '" stroke="var(--axis)" stroke-width="1" opacity="0"/>';
    el.innerHTML = legendHtml(series.map((s, i) => ({ name: s.name, color: SERIES[i] }))) +
      '<svg class="viz" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(opt.label) + '">' + g +
      '<rect x="' + pad.l + '" y="' + pad.t + '" width="' + (W - pad.l - pad.r) + '" height="' + (H - pad.t - pad.b) + '" fill="transparent" class="hit"/></svg>';
    const svg = el.querySelector('svg'), xh = svg.querySelector('.xh'), hit = svg.querySelector('.hit');
    hit.addEventListener('pointermove', (ev) => {
      const r = svg.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W;
      let best = 0;
      cats.forEach((c, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
      xh.setAttribute('x1', x(best)); xh.setAttribute('x2', x(best)); xh.setAttribute('opacity', '.6');
      showTip(ev, opt.tipTitle ? opt.tipTitle(cats[best]) : String(cats[best]), series.map((s, i) => ({ name: s.name, value: s.values[best] == null ? 'n/a' : opt.fmt(s.values[best]), color: SERIES[i] })));
    });
    hit.addEventListener('pointerleave', () => { xh.setAttribute('opacity', '0'); hideTip(); });
  }

  // horizontal bars, one series, sorted by the caller
  function barChart(el, rows, opt) {
    const W = 460, rowH = 22, pad = { l: opt.labelW || 170, r: 56, t: 4, b: 4 };
    const H = pad.t + pad.b + rows.length * rowH;
    const max = Math.max.apply(null, rows.map((r) => Math.max(0, r.value)).concat([opt.max || 0, 1e-9]));
    const w = (v) => (Math.max(0, v) / max) * (W - pad.l - pad.r);
    let g = '';
    rows.forEach((r, i) => {
      const yy = pad.t + i * rowH;
      g += '<g class="bar" data-i="' + i + '" tabindex="0"><rect x="0" y="' + yy + '" width="' + W + '" height="' + rowH + '" fill="transparent"/>' +
        '<text x="' + (pad.l - 8) + '" y="' + (yy + 15) + '" text-anchor="end">' + esc(r.label) + '</text>' +
        '<rect x="' + pad.l + '" y="' + (yy + 4) + '" width="' + Math.max(2, w(r.value)).toFixed(1) + '" height="' + (rowH - 8) + '" rx="4" fill="' + (r.color || SERIES[0]) + '"/>' +
        '<text class="lbl" x="' + (pad.l + Math.max(2, w(r.value)) + 6) + '" y="' + (yy + 15) + '">' + esc(opt.fmt(r.value)) + '</text></g>';
    });
    el.innerHTML = '<svg class="viz" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(opt.label) + '">' + g + '</svg>';
    el.querySelectorAll('g.bar').forEach((b) => {
      const r = rows[Number(b.dataset.i)];
      const on = (ev) => { b.style.opacity = '.85'; showTip(ev, r.label, (r.tip || [{ name: opt.name || 'value', value: opt.fmt(r.value) }])); };
      b.addEventListener('pointermove', on);
      b.addEventListener('focus', (ev) => { const bb = b.getBoundingClientRect(); on({ clientX: bb.right - 60, clientY: bb.top }); });
      b.addEventListener('pointerleave', () => { b.style.opacity = ''; hideTip(); });
      b.addEventListener('blur', hideTip);
    });
  }

  // predicted vs actual, one panel; nearest point gets the tooltip
  function scatter(el, pts, key, color, title, high) {
    const W = 460, H = 330, pad = { l: 40, r: 10, t: 22, b: 30 };
    const vals = pts.flatMap((p) => [p.y, p[key]]);
    const lo = Math.min(-0.05, Math.min.apply(null, vals)), hi = Math.max(0.3, Math.max.apply(null, vals));
    const x = (v) => pad.l + ((v - lo) / (hi - lo)) * (W - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
    let g = '<text class="lbl" x="' + pad.l + '" y="13">' + esc(title) + '</text>';
    [0, 0.1, 0.2, 0.3].filter((v) => v <= hi).forEach((v) => {
      g += '<line x1="' + pad.l + '" x2="' + (W - pad.r) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="var(--grid)"/><text x="' + (pad.l - 5) + '" y="' + (y(v) + 3.5) + '" text-anchor="end">' + v.toFixed(1) + '</text>' +
        '<text x="' + x(v) + '" y="' + (H - 16) + '" text-anchor="middle">' + v.toFixed(1) + '</text>';
    });
    g += '<line x1="' + x(lo) + '" y1="' + y(lo) + '" x2="' + x(hi) + '" y2="' + y(hi) + '" stroke="var(--axis)" stroke-dasharray="3 3" opacity=".7"/>';
    g += '<line x1="' + x(high) + '" x2="' + x(high) + '" y1="' + pad.t + '" y2="' + (H - pad.b) + '" stroke="#e0472d" stroke-dasharray="4 3" opacity=".7"/>' +
      '<line x1="' + pad.l + '" x2="' + (W - pad.r) + '" y1="' + y(high) + '" y2="' + y(high) + '" stroke="#e0472d" stroke-dasharray="4 3" opacity=".7"/>';
    pts.forEach((p) => { g += '<circle cx="' + x(p[key]).toFixed(1) + '" cy="' + y(p.y).toFixed(1) + '" r="3" fill="' + color + '" fill-opacity=".45"/>'; });
    g += '<text x="' + ((W + pad.l) / 2) + '" y="' + (H - 2) + '" text-anchor="middle">predicted a week ahead</text>';
    el.innerHTML = '<svg class="viz" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(title) + ': predicted against what the satellite saw">' + g + '</svg>';
    const svg = el.querySelector('svg');
    svg.addEventListener('pointermove', (ev) => {
      const r = svg.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W, py = ((ev.clientY - r.top) / r.height) * H;
      let best = null, bd = 18 * 18;
      pts.forEach((p) => { const d = (x(p[key]) - px) ** 2 + (y(p.y) - py) ** 2; if (d < bd) { bd = d; best = p; } });
      if (!best) { hideTip(); return; }
      showTip(ev, title, [{ name: 'predicted', value: fmt2(best[key]), color: color }, { name: 'satellite saw', value: fmt2(best.y) }, { name: 'water body', value: best.site }]);
    });
    svg.addEventListener('pointerleave', hideTip);
  }

  // ====================================================================== science mode
  function openSci(open) {
    $('#sciModal').hidden = !open;
    if (open) loadScience();
  }
  $('#btnSci').addEventListener('click', () => openSci(true));
  $('#sciClose').addEventListener('click', () => openSci(false));
  $('#sciModal').addEventListener('click', (e) => { if (e.target.id === 'sciModal') openSci(false); });

  async function loadScience() {
    const { ok, data } = await api('/api/science');
    if (!ok) { $('#sciBody').innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:14px">' + esc(data.error || 'Could not load.') + '</p>'; return; }
    renderScience(data);
  }

  function renderScience(d) {
    const rep = d.report || {}, bm = rep.by_method || {}, m = bm.model || {}, p = bm.persistence || {};
    const inUse = d.method === 'model' ? 'Trained model' : 'Latest reading carried forward';
    let h = '<div class="headline">Alerts use: ' + esc(inUse) + '. ' + esc(d.method_reason) + '</div>';
    if (d.data_source === 'synthetic') h += '<div class="data-banner inline">SYNTHETIC DEMO DATA: these numbers describe made-up data.</div>';
    h += '<div class="tiles">' +
      tile('F1 score, trained model', fmt2(m.f1), 'catching high readings; higher is better') +
      tile('F1 score, latest reading', fmt2(p.f1), 'the simple rule the model must beat') +
      tile('Ranking (AUC), model / rule', fmt2(m.auc) + ' / ' + fmt2(p.auc), '0.5 is a coin flip, 1.0 is perfect') +
      tile('High readings caught', fmtPct(m.recall) + ' / ' + fmtPct(p.recall), 'model / simple rule') + '</div>';
    h += '<p class="muted small" style="margin-top:8px">Tested year by year on ' + esc((rep.test_years || []).join(', ')) + ' (' + esc(rep.n_test_rows) + ' readings), each year using only earlier years to train. ' +
      esc(rep.n_actual_high) + ' readings were actually high, on only ' + esc(rep.n_actual_high_dates) + ' dates, so treat small differences as noise. Trained on ' + esc(d.trained_rows) + ' rows up to ' + esc(d.trained_through) + '.</p>';

    h += '<div class="sci-grid">' +
      '<section class="sci-section"><h3>Average error by test year</h3><p class="muted small">How far each forecast was from the next satellite reading (lower is better).</p><div id="sciYears"></div><details class="tableview"><summary>Table</summary><div id="sciYearsT"></div></details></section>' +
      '<section class="sci-section"><h3>Track record of the alerts</h3><p class="muted small">For the rule the alerts use (' + esc(inUse.toLowerCase()) + '): after each tier, how often the reading a week later really was high, in years it was never tuned on.</p><div id="sciTrack"></div></section>' +
      '</div>';
    h += '<section class="sci-section"><h3>Predicted against what the satellite saw</h3><p class="muted small">Each dot is one zone on one date, in a year the method never trained on (a random ' + esc((d.scatter || []).length) + ' of ' + esc(d.scatter_total || (d.scatter || []).length) + ' test points, so the plot stays readable). Dots on the dashed diagonal are perfect; red lines mark where the flag starts. The trained model bunches in the middle, which is why it rarely raises an alert.</p>' +
      '<div class="sci-grid"><div id="sciScatterM"></div><div id="sciScatterP"></div></div></section>';
    const hz = d.horizons && d.horizons.summary && d.horizons.summary.days && d.horizons.summary.days.length ? d.horizons : null;
    h += '<section class="sci-section"><h3>How far ahead can we see?</h3>' + (hz
      ? '<p class="muted small">Score at catching high readings 1 to 7 days ahead, in seasons each method never trained on. For the ML, each day uses the best of 15 models (5 ideas x 3 settings) found by ' + esc(hz.runner) + ' (' + esc(hz.n) + ' experiments, ' + esc(hz.finished_at) + ').</p><div id="sciAhead"></div><div id="sciAheadT"></div>'
      : '<p class="muted small">Not trained yet: Forecast Lab (L) > Train 1-7 day forecasts.</p>') + '</section>';
    h += '<section class="sci-section"><h3>Which inputs the model relies on</h3><div id="sciImp"></div></section>';
    h += '<section class="sci-section"><h3>Ideas tested in Daytona</h3><div id="sciIdeas"></div></section>';
    h += '<section class="sci-section"><h3>Water bodies in the data</h3><div id="sciSites"></div></section>';
    h += '<section class="sci-section"><h3>What today\'s forecast is built from</h3><p class="muted small">The inputs for the selected area (' + esc(selectedName()) + '), as the model sees them.</p><div id="sciInputs"></div></section>';
    $('#sciBody').innerHTML = h;

    if (hz) {
      const dd = hz.summary.days;
      lineChart($('#sciAhead'), dd.map((x) => '+' + x.days), [
        { name: 'Latest reading carried forward', short: 'latest', values: dd.map((x) => x.rule_f1) },
        { name: 'Best ML model for that day', short: 'ML', values: dd.map((x) => x.ml_f1) }],
        { label: 'F1 score by days ahead for the simple rule and the best ML model', fmt: (v) => v.toFixed(2), decimals: 2, tipTitle: (c) => c + ' days ahead' });
      $('#sciAheadT').innerHTML = trainHtml(hz);
    }

    // error by year
    const years = Object.keys(rep.by_year || {});
    if (years.length) {
      const ser = [{ name: 'Trained model', short: 'model', values: years.map((y) => rep.by_year[y].mae_model) },
        { name: 'Latest reading carried forward', short: 'latest', values: years.map((y) => rep.by_year[y].mae_persistence) },
        { name: 'Seasonal average', short: 'seasonal', values: years.map((y) => rep.by_year[y].mae_climatology) }];
      lineChart($('#sciYears'), years, ser, { label: 'Average error by test year for three methods', fmt: (v) => v.toFixed(3), tipTitle: (y) => 'Test year ' + y });
      $('#sciYearsT').innerHTML = '<table class="data"><tr><th>Year</th><th class="num">Readings</th>' + ser.map((s) => '<th class="num">' + esc(s.name) + '</th>').join('') + '</tr>' +
        years.map((y, i) => '<tr><td>' + esc(y) + '</td><td class="num">' + esc(rep.by_year[y].n) + '</td>' + ser.map((s) => '<td class="num">' + s.values[i].toFixed(4) + '</td>').join('') + '</tr>').join('') + '</table>';
    } else $('#sciYears').innerHTML = '<p class="muted small">Not enough history.</p>';

    // track record
    const tr = d.tier_record || {};
    const trRows = ['high', 'elevated', 'low'].filter((t) => tr[t] && tr[t].n).map((t) => ({
      label: 'Forecast ' + (t === 'high' ? 'flagged' : t), value: tr[t].next_high, color: t === 'high' ? '#e0472d' : t === 'elevated' ? '#e29a14' : SERIES[0],
      tip: [{ name: 'next reading high', value: fmtPct(tr[t].next_high) }, { name: 'at least elevated', value: fmtPct(tr[t].next_elevated_or_high) }, { name: 'forecasts', value: String(tr[t].n) }],
    }));
    if (trRows.length) barChart($('#sciTrack'), trRows, { label: 'Share of forecasts in each tier followed by a high reading', fmt: fmtPct, max: 1, labelW: 130 });

    // scatter, small multiples
    const sc = d.scatter || [];
    if (sc.length) {
      scatter($('#sciScatterM'), sc, 'model', SERIES[0], 'Trained model', d.thresholds.high);
      scatter($('#sciScatterP'), sc, 'persist', SERIES[1], 'Latest reading carried forward', d.thresholds.high);
    }

    // importance + ideas from the latest lab run
    const lab = d.lab && d.lab.summary;
    const imp = lab && lab.importance;
    if (imp && imp.importance) {
      $('#sciImp').innerHTML = '<div class="headline" style="margin-top:6px">' + esc(importanceTakeaway(imp)) + '</div>' +
        (imp.groups ? '<h4 style="margin:12px 0 2px">By kind of input</h4><p class="muted small">Each group is scrambled as a whole, so inputs that overlap (the latest reading and the mean of the last 3) cannot cover for each other.</p><div id="sciImpGroups"></div>' : '') +
        '<h4 style="margin:12px 0 2px">One input at a time</h4>' +
        '<p class="muted small">' + esc(imp.note) + ' This describes the trained model, which the alerts do not use (they use the latest reading). Run by: ' + esc(d.lab.runner || 'unknown') + ', ' + esc(d.lab.finished_at || '') + '.</p><div id="sciImpChart"></div>';
      if (imp.groups) {
        barChart($('#sciImpGroups'), imp.groups.map((g) => ({ label: g.group, value: g.drop, tip: [{ name: 'drop in ranking score', value: g.drop.toFixed(3) }, { name: 'inputs', value: String(g.inputs.length) }] })),
          { label: 'Drop in average precision when each group of inputs is scrambled together', fmt: (v) => (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3) });
      }
      barChart($('#sciImpChart'), imp.importance.slice(0, 12).map((r) => ({ label: r.label, value: r.drop, tip: [{ name: 'drop in ranking score', value: r.drop.toFixed(3) }, { name: 'spread across years', value: '±' + r.spread.toFixed(3) }] })),
        { label: 'Drop in average precision when each input is scrambled', fmt: (v) => (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3) });
    } else $('#sciImp').innerHTML = '<p class="muted small">Run the Forecast Lab once: one of its sandboxes measures this.</p>';
    if (lab) {
      let ih = '<div class="headline" style="margin-top:6px">' + esc(lab.headline) + '</div>';
      (lab.confirmations || []).forEach((c) => {
        ih += '<p class="small" style="margin-top:8px"><b>' + c.horizon + ' days ahead:</b> the best idea, ' + esc(c.name) + ', ' + esc(c.verdict) + ' (score ' + fmt2(c.f1) + ' vs ' + fmt2(c.base_f1) + ' for the simple rule).</p>';
      });
      Object.keys(lab.ideas || {}).forEach((k) => { ih += '<div class="idea"><b>' + esc(METHOD_NAMES[k] || k) + '</b><span class="muted">' + esc(lab.ideas[k]) + '</span></div>'; });
      $('#sciIdeas').innerHTML = ih;
    } else $('#sciIdeas').innerHTML = '<p class="muted small">No Forecast Lab run yet.</p>';

    // sites
    $('#sciSites').innerHTML = '<table class="data"><tr><th>Water body</th><th>Type</th><th class="num">Zones</th><th class="num">Readings</th><th>Since</th><th class="num">Median index</th><th class="num">Readings flagged</th><th class="num">People within 1 km</th></tr>' +
      (d.sites || []).map((s) => '<tr><td>' + esc(s.name) + '</td><td>' + esc(s.kind || '') + '</td><td class="num">' + s.zones + '</td><td class="num">' + s.readings + '</td><td>' + esc(s.first) + '</td><td class="num">' + s.median.toFixed(3) +
        '</td><td class="num">' + fmtPct(s.flagged_share) + '</td><td class="num">' + (s.people == null ? 'n/a' : s.people.toLocaleString()) + '</td></tr>').join('') + '</table>' +
      '<p class="muted small" style="margin-top:6px">' + esc(d.population_note || 'Population: run scripts/08_fetch_population.py.') + ' Source: US Census Bureau (ACS 2022).</p>';

    // inputs for the selected zone
    const z = S && S.zones.find((x) => x.id === selected);
    const labels = {};
    (d.features || []).forEach((f) => { labels[f.id] = f.label; });
    if (z && z.inputs) {
      $('#sciInputs').innerHTML = '<table class="data">' + Object.keys(z.inputs).map((k) => '<tr><td>' + esc(labels[k] || k) + '</td><td class="muted">' + esc(k) + '</td><td class="num">' + z.inputs[k] + '</td></tr>').join('') + '</table>';
    } else $('#sciInputs').innerHTML = '<p class="muted small">' + esc(z ? (z.reason || 'No forecast for this area right now.') : 'Select an area on the map.') + '</p>';
  }
  // one plain sentence: what the importance numbers say, in words
  function importanceTakeaway(imp) {
    const r = imp.importance.filter((x) => x.drop > 0);
    if (!r.length) return 'No single input stood out: scrambling any one of them barely changed the model.';
    let t = 'The model relies most on ' + r[0].label.toLowerCase() + ': scrambling it hurts';
    t += r.length > 1 && r[1].drop > 0 ? ' about ' + (r[0].drop / r[1].drop).toFixed(1) + '× more than the next input (' + r[1].label.toLowerCase() + ').' : ' far more than any other input.';
    const g = imp.groups || [];
    if (g.length > 1 && g[0].drop > 0) {
      const w = g.find((x) => x.group === 'Weather');
      t += ' By kind, ' + g[0].group.toLowerCase() + ' matter most' + (w && g[0].group !== 'Weather' ? (w.drop <= 0.2 * g[0].drop ? '; weather adds little.' : '; weather adds some.') : '.');
    }
    return t;
  }
  function tile(k, v, s) { return '<div class="tile"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div><div class="s">' + esc(s) + '</div></div>'; }
  function selectedName() { const z = S && S.zones.find((x) => x.id === selected); return z ? z.name : 'none'; }

  // ====================================================================== cleanup planner (costs, Census, Nessie, Grok)
  let planLast = null;
  async function openPlan(open) {
    $('#planModal').hidden = !open;
    if (!open) return;
    if (!$('#planSite').options.length) {
      const { data } = await api('/api/plan/sites');
      $('#planSite').innerHTML = (Array.isArray(data) ? data : []).map((s) => '<option value="' + esc(s.key) + '">' + esc(s.name) + '</option>').join('');
      const z = S && S.zones.find((x) => x.id === selected);
      if (z && z.site) $('#planSite').value = z.site;
      $('#planBudget').value = prefs.get('planBudget', '');
    }
    loadPlan();
  }
  $('#btnPlan').addEventListener('click', () => openPlan(true));
  $('#planClose').addEventListener('click', () => openPlan(false));
  $('#planModal').addEventListener('click', (e) => { if (e.target.id === 'planModal') openPlan(false); });
  $('#planForm').addEventListener('submit', (e) => { e.preventDefault(); prefs.set('planBudget', $('#planBudget').value); loadPlan(); });
  $('#planSite').addEventListener('change', loadPlan);

  async function loadPlan() {
    const site = $('#planSite').value;
    if (!site) return;
    const qs = '?site=' + encodeURIComponent(site) + '&budget=' + encodeURIComponent($('#planBudget').value || '') + '&lbs=' + encodeURIComponent($('#planLbs').value || '');
    const { ok, data } = await api('/api/plan' + qs);
    if (!ok) { $('#planBody').innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:12px">' + esc(data.error || 'Could not load.') + '</p>'; return; }
    planLast = data;
    renderPlan(data);
    api('/api/plan/businesses?site=' + encodeURIComponent(site)).then(({ data: b }) => { if (planLast === data) renderBusinesses(b); });
  }

  function renderPlan(p) {
    const f = p.facts, budget = p.budget;
    let h = '<div class="tiles">' +
      tile('Open water (satellite)', f.acres.toLocaleString() + ' acres', f.area_km2 + ' km², away from the banks') +
      tile('People within 1 km', f.people == null ? 'n/a' : f.people.toLocaleString(), f.density_per_km2 ? f.density_per_km2.toLocaleString() + ' per km² (Census ACS 2022)' : 'run scripts/08_fetch_population.py') +
      tile('Readings flagged', fmtPct(f.flagged_share), f.n_readings + ' satellite readings since 2019') +
      tile('Type', f.kind === 'river' ? 'River' : 'Lake or pond', f.drinking_water ? 'includes a drinking-water reservoir' : '') + '</div>';
    h += '<div class="headline">' + esc(p.recommendation) + '</div>';
    // cost axis: log scale, shared by every option so bars compare
    const costs = p.options.filter((o) => o.cost_10y).flatMap((o) => o.cost_10y).concat(budget ? [budget] : []);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.max(1000, Math.min.apply(null, costs))))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, costs))));
    const pos = (v) => ((Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * 100;
    h += '<p class="muted small" style="margin-top:10px">Bars show the 10-year cost range on a log scale from ' + fmtUsd(lo) + ' to ' + fmtUsd(hi) + (budget ? '; the black line is your budget' : '') + '. Fit is for a ' + (f.kind === 'river' ? 'river' : 'lake') + '.</p>';
    p.options.forEach((o) => {
      const bar = o.cost_10y
        ? '<div class="costbar" title="' + esc(fmtUsd(o.cost_10y[0]) + ' to ' + fmtUsd(o.cost_10y[1])) + '"><div class="track"></div><div class="span" style="left:' + pos(o.cost_10y[0]).toFixed(1) + '%;width:' + Math.max(1, pos(o.cost_10y[1]) - pos(o.cost_10y[0])).toFixed(1) + '%"></div>' +
          (budget ? '<div class="budget" style="left:' + pos(budget).toFixed(1) + '%"></div>' : '') + '</div><div class="costnums">' + fmtUsd(o.cost_10y[0]) + ' to ' + fmtUsd(o.cost_10y[1]) + ' over ' + p.years + ' years' +
          (o.per_person_10y ? ' &middot; $' + o.per_person_10y[0].toLocaleString() + ' to $' + o.per_person_10y[1].toLocaleString() + ' per nearby resident' : '') +
          (o.within_budget ? ' &middot; within budget: <b>' + esc(o.within_budget) + '</b>' : '') + '</div>'
        : '<div class="costnums">No published price: get an applicator\'s quote.</div>';
      const commit = o.cost_10y && p.ledger && p.ledger.funded ? '<button class="btn small-btn" data-commit="' + esc(o.id) + '" data-amount="' + o.cost_10y[0] + '" title="Record the low estimate as a payment from this water body\'s fund">Fund at ' + fmtUsd(o.cost_10y[0]) + '</button>' : '';
      h += '<div class="opt"><div><h4>' + esc(o.name) + '<span class="fit ' + esc(o.fit) + '">' + esc(o.fit) + '</span>' + (o.id === p.recommended ? '<span class="fit good">recommended</span>' : '') + '</h4>' +
        '<div class="what">' + esc(o.what) + ' ' + esc(o.fit_why) + '</div>' + (o.warning ? '<div class="warnline">' + esc(o.warning) + '</div>' : '') +
        '<div class="src">' + esc(o.cost_text) + ' ' + o.sources.map((s) => '<a href="' + esc(s[1]) + '" target="_blank" rel="noopener">' + esc(s[0]) + '</a>').join(' &middot; ') + '</div></div>' +
        '<div>' + bar + '</div><div>' + commit + '</div></div>';
    });
    h += '<p class="muted small" style="margin-top:10px">' + esc(p.caution) + '</p>';
    // ledger
    const L = p.ledger || {};
    h += '<div class="ledger"><b>Cleanup fund</b> <span class="chip' + (L.backend === 'nessie' ? '' : ' warn') + '">' + esc(L.label || '') + '</span>';
    if (L.error) h += '<p class="warnline">' + esc(L.error) + '</p>';
    else {
      h += '<p style="margin-top:6px">Funded ' + fmtUsd(L.funded) + ' &middot; committed ' + fmtUsd(L.spent) + ' &middot; <b>left ' + fmtUsd(L.remaining) + '</b>' + (L.account_id ? ' <span class="muted small">(Nessie account ' + esc(L.account_id) + ')</span>' : '') + '</p>';
      (L.transactions || []).forEach((t) => { h += '<div class="tx">' + esc(t.date) + ': ' + fmtUsd(t.amount) + ' for ' + esc(t.memo) + (t.status ? ' (' + esc(t.status) + ')' : '') + '</div>'; });
      h += '<p style="margin-top:8px"><button class="btn small-btn" id="planFund"' + (budget ? '' : ' disabled title="Enter a budget above first"') + '>Put ' + (budget ? fmtUsd(budget) : 'the budget') + ' into this fund</button></p>';
    }
    h += '</div><h3>Businesses near the water</h3><div id="planBiz"><p class="muted small">Looking up...</p></div>';
    $('#planBody').innerHTML = h;
    const fund = $('#planFund');
    if (fund) fund.addEventListener('click', async () => {
      const r = await api('/api/plan/fund', { site: p.facts.site, amount: budget });
      if (!r.ok) { toast(r.data.error || 'Could not add the budget.'); return; }
      toast('Added ' + fmtUsd(budget) + ' to the ' + p.facts.name + ' fund (' + r.data.label + ').');
      loadPlan();
    });
    $('#planBody').querySelectorAll('[data-commit]').forEach((b) => b.addEventListener('click', async () => {
      const r = await api('/api/plan/spend', { site: p.facts.site, option: b.dataset.commit, amount: Number(b.dataset.amount) });
      if (!r.ok) { toast(r.data.error || 'Could not record that.'); return; }
      toast('Recorded in the ' + r.data.label + '. ' + fmtUsd(r.data.remaining) + ' left.');
      loadPlan();
    }));
  }

  function renderBusinesses(b) {
    const box = $('#planBiz');
    if (!box) return;
    if (!b || !b.ok) { box.innerHTML = '<p class="muted small">' + esc((b && (b.reason || b.error)) || 'Not available.') + '</p>'; return; }
    if (!b.merchants.length) { box.innerHTML = '<p class="muted small">Nessie lists no merchants within ' + b.radius_miles + ' mile of this water. ' + esc(b.note) + '</p>'; return; }
    box.innerHTML = '<p class="muted small">' + b.count + ' within ' + b.radius_miles + ' mile. ' + esc(b.note) + '</p><ul class="biz">' +
      b.merchants.map((m) => '<li>' + esc(m.name) + (m.category ? ' <span class="muted">(' + esc(m.category) + ')</span>' : '') + '</li>').join('') + '</ul>';
  }

  $('#planAskForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('#planAskInput').value.trim();
    if (!q) return;
    const out = $('#planAskResult');
    out.innerHTML = '<p class="muted small" style="margin-top:10px">Thinking...</p>';
    const r = await api('/api/plan/ask', { site: $('#planSite').value, question: q, budget: $('#planBudget').value || null });
    if (!r.ok) { out.innerHTML = '<p class="small" style="color:var(--alert-ink);margin-top:10px">' + esc(r.data.error || 'Could not answer.') + '</p>'; return; }
    out.innerHTML = '<div class="outbox" style="margin-top:10px"></div><p class="muted small" style="margin-top:4px">' + (r.data.source === 'grok' ? 'Grok, using only the computed table above.' : 'Fixed comparison (Grok not connected).') + '</p>';
    out.querySelector('.outbox').textContent = r.data.reply;
  });

  // ====================================================================== photo review (this computer only) and story bubbles
  const BLOOM_WORDS = { bloom: 'Looks like a bloom', no_bloom: 'No bloom visible', unclear: "Can't tell from this photo" };
  const STATUS_TABS = [['review', 'Needs review'], ['pending', 'Not screened'], ['verified', 'Verified'], ['rejected', 'Rejected']];
  let photoTab = 'review';
  function openPhotos(open) {
    $('#photoModal').hidden = !open;
    if (!open) return;
    const c = (S && S.photo_counts) || {};
    photoTab = ['review', 'pending', 'verified', 'rejected'].find((k) => c[k]) || 'review';   // open where there is something to see
    loadQueue();
  }
  $('#btnPhotos').addEventListener('click', () => openPhotos(true));
  $('#photoClose').addEventListener('click', () => openPhotos(false));
  $('#photoModal').addEventListener('click', (e) => { if (e.target.id === 'photoModal') openPhotos(false); });

  function renderPhotoBadge() {
    const c = (S && S.photo_counts) || {};
    const n = (c.review || 0) + (c.pending || 0);
    const b = $('#photoBadge');
    b.hidden = !n;
    b.textContent = String(n);
  }

  async function loadQueue() {
    const { ok, data } = await api('/api/photos/queue?status=' + photoTab);
    if (!ok) { $('#photoList').innerHTML = '<p class="small" style="color:var(--alert-ink)">' + esc(data.error || 'Could not load.') + '</p>'; return; }
    const c = data.counts || {};
    $('#photoTabs').innerHTML = STATUS_TABS.map(([k, label]) => '<button type="button" data-tab="' + k + '" aria-pressed="' + (k === photoTab) + '">' + esc(label) + ' <span class="count">' + (c[k] || 0) + '</span></button>').join('');
    const caps = (S && S.capabilities) || {};
    $('#photoScreen').disabled = !caps.vision || !(c.pending);
    $('#photoScreen').title = caps.vision ? 'Ask Grok about every photo that has not been screened' : 'Needs XAI_API_KEY in .env';
    if (!data.photos.length) { $('#photoList').innerHTML = '<p class="muted small" style="margin-top:12px">Nothing here.</p>'; return; }
    $('#photoList').innerHTML = data.photos.map((p) => {
      const acts = [];
      if (p.status !== 'verified') acts.push('<button class="btn small-btn primary" data-act="verify">Verify</button>');
      if (p.status !== 'rejected') acts.push('<button class="btn small-btn" data-act="reject">Reject</button>');
      if (p.status === 'verified' || p.status === 'rejected') acts.push('<button class="btn small-btn" data-act="review">Back to review</button>');
      acts.push('<button class="btn small-btn danger" data-act="delete">Delete</button>');
      return '<div class="pcard" data-id="' + p.id + '"><img src="/photos/' + p.id + '.jpg" alt="Photo sent by ' + esc(p.from) + '" loading="lazy">' +
        '<div class="pmeta"><div><b>' + esc(p.zone_name || 'Not placed yet') + '</b>' + (p.site_name ? ' <span class="muted">' + esc(p.site_name) + '</span>' : '') + '</div>' +
        '<div class="muted small">' + esc(new Date(p.at).toLocaleString()) + ' &middot; from ' + esc(p.from) + '</div>' +
        (p.bloom ? '<span class="fit ' + (p.bloom === 'bloom' ? 'risky' : p.bloom === 'no_bloom' ? 'good' : 'uncertain') + '">' + esc(BLOOM_WORDS[p.bloom] || p.bloom) + '</span>' : '') +
        (p.description ? '<p class="small" style="margin-top:4px">' + esc(p.description) + '</p>' : '') +
        '<p class="muted small" style="margin-top:4px">' + esc(p.decided_by === 'manual' ? 'Set by hand' : p.decided_by === 'grok' ? 'Grok: ' + (p.reason || '') : (p.reason || 'Not screened yet')) + '</p>' +
        '<div class="pacts">' + acts.join('') + '</div></div></div>';
    }).join('');
  }
  $('#photoTabs').addEventListener('click', (e) => { const b = e.target.closest('[data-tab]'); if (b) { photoTab = b.dataset.tab; loadQueue(); } });
  $('#photoList').addEventListener('click', async (e) => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const id = Number(b.closest('.pcard').dataset.id);
    if (b.dataset.act === 'delete' && !window.confirm('Delete this photo for good? This cannot be undone.')) return;
    const r = await api('/api/photos/moderate', { id: id, action: b.dataset.act });
    if (!r.ok) { toast(r.data.error || 'Could not change that photo.'); return; }
    toast(b.dataset.act === 'delete' ? 'Photo deleted.' : 'Photo moved to ' + r.data.status + '.');
    loadQueue(); refresh(); loadStories();
  });
  $('#photoScreen').addEventListener('click', async () => {
    const r = await api('/api/photos/screen', {});
    if (!r.ok) { toast(r.data.error || 'Could not start.'); return; }
    $('#photoScreen').disabled = true;
    toast('Asking Grok about each photo (a few seconds each)...');
    const done = await pollJob('photos', r.data.job_id, () => {});
    if (done.status === 'done' && done.result) {
      const c = done.result;
      toast('Screened: ' + c.verified + ' verified, ' + c.rejected + ' rejected, ' + c.review + ' need review.');
    } else toast(done.error || 'Screening stopped.');
    loadQueue(); refresh(); loadStories();
  });

  // ---- story bubbles: verified photos from the last 7 days, one bubble per area; red ring = looks like a bloom
  const storyMarkers = {};
  let storiesOn = prefs.get('stories', true);
  async function loadStories() {
    const { ok, data } = await api('/api/stories');
    if (!ok || !Array.isArray(data)) return;
    const ids = new Set(data.map((g) => g.zone));
    Object.keys(storyMarkers).forEach((id) => { if (!ids.has(id) || !storiesOn) { storyMarkers[id].remove(); delete storyMarkers[id]; } });
    if (!storiesOn) return;
    data.forEach((g) => {
      if (storyMarkers[g.zone]) storyMarkers[g.zone].remove();
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'story' + (g.bloom ? ' bloom' : '');
      el.setAttribute('aria-label', g.photos.length + ' community photo' + (g.photos.length === 1 ? '' : 's') + ' near ' + g.name + (g.bloom ? ', at least one looks like a bloom' : ''));
      const img = document.createElement('img');
      img.src = '/photos/' + g.photos[0].id + '.jpg';
      img.alt = '';
      el.appendChild(img);
      if (g.photos.length > 1) { const n = document.createElement('span'); n.className = 'n'; n.textContent = String(g.photos.length); el.appendChild(n); }
      el.addEventListener('click', (ev) => { ev.stopPropagation(); openStory(g, 0); });
      // sit just above the area's marker so it never hides the red flag
      storyMarkers[g.zone] = new maplibregl.Marker({ element: el, anchor: 'bottom', offset: [0, -34] }).setLngLat([g.anchor[1], g.anchor[0]]).addTo(map);
    });
  }

  let story = null;
  function openStory(g, i) {
    story = { g: g, i: i };
    const p = g.photos[i];
    $('#storyImg').src = '/photos/' + p.id + '.jpg';
    $('#storyTitle').textContent = g.name + (g.site_name ? ' · ' + g.site_name : '');
    $('#storyWhen').textContent = new Date(p.at).toLocaleString() + (g.photos.length > 1 ? '  (' + (i + 1) + ' of ' + g.photos.length + ')' : '');
    const lab = $('#storyLabel');
    lab.textContent = 'Grok: ' + (p.label || BLOOM_WORDS.unclear);
    lab.className = 'fit ' + (p.bloom === 'bloom' ? 'risky' : p.bloom === 'no_bloom' ? 'good' : 'uncertain');
    $('#storyDesc').textContent = p.description || '';
    $('#storyBar').innerHTML = g.photos.map((_, k) => '<span class="' + (k <= i ? 'on' : '') + '"></span>').join('');
    $('#storyPrev').disabled = i === 0;
    $('#storyNext').disabled = i === g.photos.length - 1;
    $('#story').hidden = false;
  }
  function closeStory() { $('#story').hidden = true; story = null; }
  $('#storyClose').addEventListener('click', closeStory);
  $('#story').addEventListener('click', (e) => { if (e.target.id === 'story') closeStory(); });
  $('#storyPrev').addEventListener('click', () => { if (story && story.i > 0) openStory(story.g, story.i - 1); });
  $('#storyNext').addEventListener('click', () => { if (story && story.i < story.g.photos.length - 1) openStory(story.g, story.i + 1); });
  $('#storyToggle').checked = storiesOn;
  $('#storyToggle').addEventListener('change', (e) => { storiesOn = e.target.checked; prefs.set('stories', storiesOn); loadStories(); });

  // ====================================================================== keyboard
  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target || {}).tagName || '');
    if (e.key === 'Escape') { openLab(false); openChat(false); openSci(false); openPlan(false); openPhotos(false); closeStory(); hideTip(); $('#lightbox').hidden = true; return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'p' || e.key === 'P') $('#btnPanel').click();
    else if (e.key === 'l' || e.key === 'L') openLab($('#labModal').hidden);
    else if (e.key === 's' || e.key === 'S') openSci($('#sciModal').hidden);
    else if (e.key === 'b' || e.key === 'B') openPlan($('#planModal').hidden);
    else if (e.key === 'v' || e.key === 'V') openPhotos($('#photoModal').hidden);
    else if (e.key === 'c' || e.key === 'C') { e.preventDefault(); openChat($('#chat').hidden); }
  });

  // ====================================================================== loop
  async function refresh() {
    const { ok, data } = await api('/api/state?days=' + reqDays());
    if (!ok) return;
    S = data;
    render();
    updateHeat();
  }
  async function init() {
    setPanel(prefs.get('panel', 'open'));
    if (prefs.get('alertsMin', false)) { $('#alerts').classList.add('min'); $('#alertsToggle').setAttribute('aria-expanded', 'false'); }
    $('#btn3d').setAttribute('aria-pressed', String(prefs.get('pitch', 52) > 10));
    const z = await api('/api/zones');
    ZG = Array.isArray(z.data) ? z.data : [];
    FR = (await api('/api/frames')).data;
    $('#yearSel').innerHTML = (FR.years || []).slice().reverse().map((y) => '<option>' + y + '</option>').join('');
    if (map.getSource('zone-pts')) map.getSource('zone-pts').setData(pointsGeojson());
    const ah = /^#ahead=([1-7])(&ml)?$/.exec(location.hash);       // demo link, e.g. #ahead=3&ml
    if (ah) { ahead = Number(ah[1]); if (ah[2]) view = 'ml'; }
    await refresh();
    await Promise.all([loadUsers(), refreshFeed()]);
    if (S && S.mode === 'replay') setSliderTo(S.as_of); else syncLiveSlider();
    fitRiver(false);
    loadStories();
    setInterval(loadStories, 30000);
    const deep = { '#science': () => openSci(true), '#planner': () => openPlan(true), '#lab': () => openLab(true), '#photos': () => openPhotos(true) }[location.hash];   // links for demos
    if (deep) deep();
    setInterval(() => { refresh(); refreshFeed(); }, 4000);
    window.__algaebro = { map: map, state: () => S };    // for automated checks
  }
  init();
})();
