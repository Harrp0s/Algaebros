"""Build one self-contained map page: scrub through a past season's weekly forecasts and compare
each with what the satellite actually showed afterwards. Needs internet in the browser for map tiles."""
from __future__ import annotations

import json
from typing import Optional

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Algaebro: Charles River bloom-risk forecast</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
 :root{--low:#3f9c6b;--elev:#e3a62d;--high:#d0455a;--unk:#8a8f98;--ink:#14202b;--paper:#f6f4ee}
 html,body{margin:0;height:100%;font-family:"Avenir Next","Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--paper)}
 #banner{padding:8px 16px;font-weight:600;display:none}
 #banner.bad{display:block;background:#b3261e;color:#fff}
 #banner.note{display:block;background:#fff3cd;color:#5c4400}
 #wrap{display:flex;height:calc(100% - 38px)}
 #map{flex:1}
 #side{width:340px;padding:16px;overflow:auto;border-left:1px solid #d9d4c7;background:#fff}
 h1{font-size:20px;margin:0 0 4px} p{margin:6px 0;line-height:1.4;font-size:14px}
 input[type=range]{width:100%} .row{display:flex;justify-content:space-between;font-size:13px}
 .chip{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:12px;font-weight:600}
 table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px} td,th{padding:3px 4px;text-align:left;border-bottom:1px solid #eee}
 .legend span{margin-right:10px;font-size:13px} .dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px;vertical-align:-1px}
 .box{background:#f3f0e7;border-radius:8px;padding:8px 10px;margin-top:10px;font-size:13px}
</style></head><body>
<div id="banner"></div>
<div id="wrap"><div id="map"></div>
<div id="side">
 <h1>Charles River bloom-risk forecast</h1>
 <p id="sub"></p>
 <input id="slider" type="range" min="0" value="0"><div class="row"><span id="d0"></span><span id="d1"></span></div>
 <p><b id="asof"></b><br><span id="target"></span></p>
 <p id="cond"></p>
 <div class="legend"><span><i class="dot" style="background:var(--low)"></i>low</span><span><i class="dot" style="background:var(--elev)"></i>elevated</span><span><i class="dot" style="background:var(--high)"></i>high</span><span><i class="dot" style="background:var(--unk)"></i>no forecast</span></div>
 <table id="tbl"></table>
 <div class="box" id="score"></div>
 <div class="box" id="quality"></div>
 <p style="font-size:12px;color:#555">Tiers are relative to this river's own past readings of a satellite chlorophyll index (top 10% = high). This is not a toxin test and not a safety determination.</p>
</div></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__, METRICS = __METRICS__;
const C = {low:"#3f9c6b", elevated:"#e3a62d", high:"#d0455a", unknown:"#8a8f98"};
const banner = document.getElementById('banner');
if (DATA.data_source === 'synthetic') { banner.className='bad'; banner.textContent='SYNTHETIC DEMO DATA: this is not real. It only shows the software works.'; }
else if (DATA.method === 'persistence') { banner.className='note'; banner.textContent='Outlook method: the latest satellite reading carried forward. A weather model was tested and did not beat it at catching HIGH readings.'; }
const map = L.map('map');
// Basemaps that need no API key, tried in order. OpenStreetMap refuses pages opened as plain files
// (no Referer header), so it is only used when the page is served from localhost (scripts/06_make_map.py --serve).
const SOURCES = [
  {name:'OpenStreetMap', url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png', opts:{maxZoom:19, attribution:'&copy; OpenStreetMap contributors'}, needsHttp:true},
  {name:'Esri', url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', opts:{maxZoom:19, attribution:'Tiles &copy; Esri'}, needsHttp:false}
].filter(s => !(s.needsHttp && location.protocol === 'file:'));
let srcIdx = -1, tileErrors = 0, tileLayer = null;
function nextSource() {
  if (tileLayer) { map.removeLayer(tileLayer); tileLayer = null; }
  srcIdx += 1; tileErrors = 0;
  if (srcIdx >= SOURCES.length) {
    const d = document.createElement('div');
    d.style.cssText = 'position:absolute;z-index:1000;left:60px;top:12px;background:#fff3cd;color:#5c4400;padding:6px 10px;border-radius:6px;font-size:13px;max-width:420px';
    d.textContent = 'Base map tiles could not load (offline, blocked, or a key is required). The zones still show. Try: python scripts/06_make_map.py --serve';
    document.getElementById('map').appendChild(d);
    return;
  }
  tileLayer = L.tileLayer(SOURCES[srcIdx].url, SOURCES[srcIdx].opts).addTo(map);
  tileLayer.on('tileerror', () => { tileErrors += 1; if (tileErrors === 4) nextSource(); });
}
nextSource();
const all = []; const layers = {};
DATA.zones.forEach(z => {
  layers[z.id] = z.points.map(p => { all.push([p[1],p[0]]); return L.circleMarker([p[1],p[0]],{radius:4,weight:0,fillOpacity:.85,fillColor:C.unknown}).addTo(map); });
  L.marker([z.lat,z.lon],{icon:L.divIcon({className:'',html:'<div style="background:#fff;border:1px solid #333;border-radius:50%;width:22px;height:22px;text-align:center;line-height:20px;font-weight:700;font-size:12px">'+z.number+'</div>',iconSize:[22,22]})}).addTo(map);
});
if (all.length) map.fitBounds(all, {padding:[30,30]});
const slider = document.getElementById('slider'); slider.max = Math.max(0, DATA.frames.length-1);
document.getElementById('sub').textContent = (DATA.frames.length ? 'Replay of the '+DATA.year+' season: a forecast every week, then what the satellite saw next.' : 'No forecasts to show.');
if (DATA.frames.length) { document.getElementById('d0').textContent = DATA.frames[0].as_of; document.getElementById('d1').textContent = DATA.frames[DATA.frames.length-1].as_of; }
const word = t => t==='high'?'HIGH':t;
function chip(t){return '<span class="chip" style="background:'+C[t]+'">'+word(t)+'</span>';}
function render(i){
  const f = DATA.frames[i]; if(!f) return;
  document.getElementById('asof').textContent = 'Forecast issued '+f.as_of;
  document.getElementById('target').textContent = 'for '+f.target_date;
  const w = f.weather||{}; document.getElementById('cond').textContent = (w.tmean_7_c!==undefined?('7-day avg '+w.tmean_7_c+' °C'):'') + ((w.flags&&w.flags.length)?(', '+w.flags.join(', ')):'');
  let rows = '<tr><th>Zone</th><th>Forecast</th><th>Satellite saw</th></tr>', hits=0, fcHigh=0, obsHigh=0, both=0;
  DATA.zones.forEach(z => {
    const r = (f.zones||{})[z.id]||{tier:'unknown'}; const v = r.verification;
    layers[z.id].forEach(m => m.setStyle({fillColor:C[r.tier]||C.unknown}));
    const pop = '<b>'+z.name+'</b><br>Outlook: '+word(r.tier)+(r.ndci_pred!==undefined?(' (index '+r.ndci_pred.toFixed(3)+')'):'')+(v?('<br>Satellite later saw: '+word(v.observed_tier)+' on '+v.observed_date):'');
    layers[z.id].forEach(m => m.bindPopup(pop));
    rows += '<tr><td>'+z.number+'</td><td>'+chip(r.tier)+'</td><td>'+(v?chip(v.observed_tier):'<span style="color:#999">n/a</span>')+'</td></tr>';
    if (r.tier==='high') fcHigh++; if (v && v.observed_tier==='high') obsHigh++; if (r.tier==='high' && v && v.observed_tier==='high') both++;
  });
  document.getElementById('tbl').innerHTML = rows;
  document.getElementById('score').innerHTML = '<b>This week:</b> forecast HIGH in '+fcHigh+' zone(s); satellite later showed HIGH in '+obsHigh+', and '+both+' of those matched.';
}
slider.addEventListener('input', () => render(+slider.value));
let start = 0, bestN = -1;
DATA.frames.forEach((f, i) => { const n = Object.values(f.zones || {}).filter(r => r.tier === 'high').length; if (n > bestN) { bestN = n; start = i; } });
slider.value = start; render(start);
const q = document.getElementById('quality');
const R = METRICS && METRICS.report;
if (R && R.by_method) {
  const used = METRICS.method, m = R.by_method[used === 'model' ? 'model' : 'persistence'], f = x => x===null||x===undefined ? 'n/a' : Math.round(x*100)+'%';
  q.innerHTML = '<b>How good is it, overall?</b> Tested on years it never saw ('+R.test_years.join(', ')+').<br>Outlook method: <b>'+(used==='model'?'trained model':'latest satellite reading carried forward')+'</b>.<br>When it flagged HIGH, '+f(m.precision)+' were HIGH a week later; it caught '+f(m.recall)+' of all HIGH readings.<br><span style="color:#555">Those '+R.n_actual_high+' HIGH readings fall on only '+(R.n_actual_high_dates||'?')+' dates, so treat these rates as rough.</span>';
} else { q.textContent = 'No out-of-sample evaluation available yet.'; }
</script></body></html>
"""


def build_map_html(replay: dict, metrics: Optional[dict] = None) -> str:
    def dump(obj) -> str:
        return json.dumps(obj, default=str).replace("</", "<\\/")

    m = metrics or {}
    slim = {"method": m.get("method"), "report": m.get("report"), "tier_record": m.get("tier_record")}
    return PAGE.replace("__DATA__", dump(replay)).replace("__METRICS__", dump(slim))
