/* Lecture du texte d'une photo, entièrement dans le téléphone : rien n'est envoyé ailleurs.
   Moteur ONNX Runtime (WebAssembly) + modèles PP-OCR (détection des zones de texte, puis lecture de chaque zone). */
(function(){
"use strict";
const S = {ort: null, det: null, rec: null, dict: null, pret: null};
const toile = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };

function script(url){ return new Promise((ok, ko) => { const s = document.createElement("script"); s.src = url; s.onload = ok; s.onerror = () => ko(new Error("script " + url)); document.head.appendChild(s); }); }
// Télécharge un fichier en suivant l'avancement (total connu d'avance : les serveurs n'annoncent pas toujours la taille)
async function fichier(url, total, suivi){
  const r = await fetch(url); if(!r.ok) throw new Error("fichier " + url + " " + r.status);
  if(!r.body || !r.body.getReader){ const b = new Uint8Array(await r.arrayBuffer()); suivi(b.length); return b; }
  const rd = r.body.getReader(), parts = []; let n = 0;
  for(;;){ const {done, value} = await rd.read(); if(done) break; parts.push(value); n += value.length; suivi(value.length); }
  const out = new Uint8Array(n); let o = 0; for(const p of parts){ out.set(p, o); o += p.length; } return out;
}
const POIDS = {wasm: 14848852, det: 4745517, rec: 10822323};
// Texte base64 (octets ASCII) → octets. Par morceaux d'1 Mo pour ne pas doubler la mémoire.
function b64(u){
  let n = u.length; while(n && (u[n - 1] === 10 || u[n - 1] === 13)) n--;
  const fin = (u[n - 1] === 61 ? 1 : 0) + (u[n - 2] === 61 ? 1 : 0), out = new Uint8Array(n / 4 * 3 - fin), dec = new TextDecoder("latin1"); let o = 0;
  for(let i = 0; i < n; i += 1048576){ const s = atob(dec.decode(u.subarray(i, Math.min(n, i + 1048576)))); for(let j = 0; j < s.length; j++) out[o++] = s.charCodeAt(j); }
  return out;
}
// opt.b64 : les deux modèles sont servis en texte (det.b64.txt, rec.b64.txt). C'est le cas dans l'aperçu Claude, qui ne sert pas les fichiers .onnx.
function charger(base, suivi, opt){
  if(S.pret) return S.pret;
  S.pret = (async () => {
    const tot = POIDS.wasm + POIDS.det + POIDS.rec; let fait = 0; const av = n => { fait += n; if(suivi) suivi(Math.min(1, fait / tot)); };
    if(typeof WebAssembly !== "object") throw new Error("wasm");
    if(!window.ort) await script(base + "ort.wasm.min.js");
    const ort = S.ort = window.ort;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    const modele = (nom, poids) => opt && opt.b64 ? fichier(base + nom + ".b64.txt", poids, n => av(n * .75)).then(b64) : fichier(base + nom + ".onnx", poids, av);
    const [wasm, det, rec, dict] = await Promise.all([fichier(base + "ort-wasm-simd-threaded.wasm", POIDS.wasm, av), modele("det", POIDS.det),
      modele("rec", POIDS.rec), fetch(base + "dict.txt").then(r => { if(!r.ok) throw new Error("dict"); return r.text(); })]);
    ort.env.wasm.wasmBinary = wasm.buffer;
    ort.env.wasm.wasmPaths = {mjs: new URL(base + "ort-wasm-simd-threaded.js", location.href).href};
    const reglages = {executionProviders: ["wasm"], graphOptimizationLevel: "all"};
    S.det = await ort.InferenceSession.create(det, reglages);
    S.rec = await ort.InferenceSession.create(rec, reglages);
    S.dict = dict.replace(/\r/g, "").split("\n"); if(S.dict[S.dict.length - 1] === "") S.dict.pop(); S.dict.push(" ");
  })();
  S.pret.catch(() => { S.pret = null; });
  return S.pret;
}

/* ---- Détection : carte de probabilité « ici il y a du texte » → boîtes orientées ---- */
async function detecter(src, cote, opt){
  opt = opt || {};
  const k = cote / Math.max(src.width, src.height);
  const w = Math.max(32, Math.round(src.width * k / 32) * 32), h = Math.max(32, Math.round(src.height * k / 32) * 32);
  const c = toile(w, h), x = c.getContext("2d", {willReadFrequently: true}); x.drawImage(src, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
  for(let i = 0, j = 0; i < n; i++, j += 4){ f[i] = (d[j + 2] / 255 - .485) / .229; f[n + i] = (d[j + 1] / 255 - .456) / .224; f[2 * n + i] = (d[j] / 255 - .406) / .225; }
  const out = await S.det.run({[S.det.inputNames[0]]: new S.ort.Tensor("float32", f, [1, 3, h, w])});
  const p = out[S.det.outputNames[0]].data;
  return boites(p, w, h, src.width / w, src.height / h, opt);
}
function boites(p, w, h, sx, sy, opt){
  const seuil = opt.seuil || .3, mini = opt.score || .5, gonfle = opt.gonfle || 1.6;
  const n = w * h, vu = new Uint8Array(n), pile = new Int32Array(n), res = [];
  for(let s = 0; s < n; s++){
    if(vu[s] || p[s] <= seuil) continue;
    let top = 0, nb = 0, sp = 0, mx = 0, my = 0; pile[top++] = s; vu[s] = 1; const px = [];
    while(top){ const q = pile[--top], qx = q % w, qy = (q - qx) / w; px.push(q); nb++; sp += p[q]; mx += qx; my += qy;
      if(qx > 0 && !vu[q - 1] && p[q - 1] > seuil){ vu[q - 1] = 1; pile[top++] = q - 1; }
      if(qx < w - 1 && !vu[q + 1] && p[q + 1] > seuil){ vu[q + 1] = 1; pile[top++] = q + 1; }
      if(qy > 0 && !vu[q - w] && p[q - w] > seuil){ vu[q - w] = 1; pile[top++] = q - w; }
      if(qy < h - 1 && !vu[q + w] && p[q + w] > seuil){ vu[q + w] = 1; pile[top++] = q + w; } }
    if(nb < 10 || sp / nb < mini) continue;
    mx /= nb; my /= nb; let sxx = 0, syy = 0, sxy = 0;
    for(const q of px){ const dx = q % w - mx, dy = ((q / w) | 0) - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
    let a = .5 * Math.atan2(2 * sxy, sxx - syy), ca = Math.cos(a), sa = Math.sin(a);
    let u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
    const etend = () => { u0 = v0 = 1e9; u1 = v1 = -1e9; for(const q of px){ const dx = q % w - mx, dy = ((q / w) | 0) - my, u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
      if(u < u0) u0 = u; if(u > u1) u1 = u; if(v < v0) v0 = v; if(v > v1) v1 = v; } };
    etend();
    // Zone presque carrée (1 ou 2 caractères) : l'orientation calculée ne veut rien dire, on la prend droite
    if((u1 - u0 + 1) / (v1 - v0 + 1) < 1.5){ a = 0; ca = 1; sa = 0; etend(); }
    const L = u1 - u0 + 1, T = v1 - v0 + 1; if(Math.min(L, T) < 3) continue;
    const g = L * T * gonfle / (2 * (L + T)), uc = (u0 + u1) / 2, vc = (v0 + v1) / 2;
    const cx = mx + uc * ca - vc * sa + .5, cy = my + uc * sa + vc * ca + .5;
    res.push({cx: cx * sx, cy: cy * sy, w: (L + 2 * g) * sx, h: (T + 2 * g) * sy, a, s: sp / nb});
  }
  return res;
}

/* ---- Lecture d'une boîte ---- */
async function lireBoite(src, b, demiTour){
  const H = 48, W = Math.max(24, Math.min(1280, Math.round(b.w / b.h * H))), k = H / b.h;
  const c = toile(W, H), x = c.getContext("2d", {willReadFrequently: true});
  x.fillStyle = "#808080"; x.fillRect(0, 0, W, H); x.imageSmoothingQuality = "high";
  x.translate(W / 2, H / 2); x.rotate(-b.a + (demiTour ? Math.PI : 0)); x.scale(k, k); x.drawImage(src, -b.cx, -b.cy);
  const d = x.getImageData(0, 0, W, H).data, n = W * H, f = new Float32Array(3 * n);
  for(let i = 0, j = 0; i < n; i++, j += 4){ f[i] = d[j + 2] / 127.5 - 1; f[n + i] = d[j + 1] / 127.5 - 1; f[2 * n + i] = d[j] / 127.5 - 1; }
  const out = await S.rec.run({[S.rec.inputNames[0]]: new S.ort.Tensor("float32", f, [1, 3, H, W])});
  const o = out[S.rec.outputNames[0]], T = o.dims[1], K = o.dims[2], v = o.data;
  let txt = "", som = 0, nb = 0, prec = 0;
  for(let t = 0; t < T; t++){ let m = 0, mv = -1; const off = t * K; for(let q = 0; q < K; q++){ const z = v[off + q]; if(z > mv){ mv = z; m = q; } }
    if(m && m !== prec){ txt += S.dict[m - 1] || ""; som += mv; nb++; } prec = m; }
  return {t: txt, c: nb ? som / nb : 0};
}
// Lit une boîte ; un texte écrit de haut en bas ou de bas en haut est essayé dans les deux sens
async function lire(src, b){
  const debout = Math.abs(b.a) > Math.PI / 4;
  let r = await lireBoite(src, b, false);
  if(debout){ const r2 = await lireBoite(src, b, true); if(r2.c * Math.sqrt(r2.t.length) > r.c * Math.sqrt(r.t.length)) r = Object.assign(r2, {tour: true}); }
  return r;
}
/* ---- Photo entière : on repère d'abord où se trouve le tableau, puis on le relit de près, bande par bande ---- */
// Laisse l'écran se redessiner entre deux étapes (la lecture occupe le téléphone par à-coups)
const pause = () => new Promise(r => { let fait = false; const ok = () => { if(!fait){ fait = true; r(); } }; setTimeout(ok, 60); if(window.requestAnimationFrame) requestAnimationFrame(() => setTimeout(ok, 0)); });
const ARRET = () => Object.assign(new Error("arret"), {code: "cancelled"});
const cadre = b => { const c = Math.abs(Math.cos(b.a)), s = Math.abs(Math.sin(b.a)), w = b.w * c + b.h * s, h = b.w * s + b.h * c; return {x0: b.cx - w / 2, y0: b.cy - h / 2, x1: b.cx + w / 2, y1: b.cy + h / 2}; };
async function analyser(src, suivi, opt){
  opt = opt || {}; const dit = (p, m) => { if(suivi) suivi(p, m); }, stop = () => { if(opt.stop && opt.stop()) throw ARRET(); };
  dit(0, "repere"); await pause();
  const b1 = await detecter(src, opt.cote1 || 1280); await pause(); stop();
  let roi = {x: 0, y: 0, w: src.width, h: src.height};
  if(b1.length >= 3){ let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for(const b of b1){ const c = cadre(b); x0 = Math.min(x0, c.x0); y0 = Math.min(y0, c.y0); x1 = Math.max(x1, c.x1); y1 = Math.max(y1, c.y1); }
    const mx = (x1 - x0) * .1 + src.width * .02, my = (y1 - y0) * .1 + src.height * .02;
    x0 = Math.max(0, x0 - mx); y0 = Math.max(0, y0 - my); x1 = Math.min(src.width, x1 + mx); y1 = Math.min(src.height, y1 + my);
    if((x1 - x0) * (y1 - y0) < .8 * src.width * src.height) roi = {x: x0, y: y0, w: x1 - x0, h: y1 - y0}; }
  const cible = opt.cote2 || 2560, k = Math.min(opt.zoomMax || 3, cible / Math.max(roi.w, roi.h));
  const W = Math.round(roi.w * k), H = Math.round(roi.h * k), hb = opt.bande || 800, rec = 176;
  const nb = H <= hb ? 1 : Math.ceil((H - rec) / (hb - rec)); let bs = [];
  for(let i = 0; i < nb; i++){
    dit(.05 + .45 * i / nb, "zones"); await pause(); stop();
    const y0 = nb === 1 ? 0 : Math.min(i * (hb - rec), H - hb), h = Math.min(hb, H);
    const c = toile(W, h), x = c.getContext("2d", {willReadFrequently: true}); x.imageSmoothingQuality = "high";
    x.drawImage(src, roi.x, roi.y + y0 / k, roi.w, h / k, 0, 0, W, h);
    const r = await detecter(c, Math.max(W, h)); c.width = c.height = 1;
    for(const b of r){ b.cx = roi.x + b.cx / k; b.cy = roi.y + (y0 + b.cy) / k; b.w /= k; b.h /= k; bs.push(b); }
    await pause();
  }
  // Une même zone vue dans deux bandes : on garde la plus grande
  bs.sort((a, b) => b.w * b.h - a.w * a.h); const garde = [];
  for(const b of bs){ const c = cadre(b); let doublon = false;
    for(const g of garde){ const d = g.c, ix = Math.min(c.x1, d.x1) - Math.max(c.x0, d.x0), iy = Math.min(c.y1, d.y1) - Math.max(c.y0, d.y0);
      if(ix > 0 && iy > 0 && ix * iy > .5 * (c.x1 - c.x0) * (c.y1 - c.y0)){ doublon = true; break; } }
    if(!doublon){ b.c = c; garde.push(b); } }
  bs = garde;
  for(let i = 0; i < bs.length; i++){ const b = bs[i], r = await lire(src, b); b.t = r.t; b.conf = r.c; if(r.tour) b.a += b.a > 0 ? -Math.PI : Math.PI;
    if(i % 6 === 0){ dit(.5 + .5 * i / bs.length, "texte"); await pause(); stop(); } }
  dit(1, "fin");
  return {roi, k, boites: bs.map(b => ({cx: b.cx, cy: b.cy, w: b.w, h: b.h, a: b.a, s: b.s, t: b.t, c: b.conf}))};
}
window.TBOCR = {charger, detecter, lire, lireBoite, analyser, toile, etat: S};
})();
/* Du texte lu sur la photo (boîtes avec position) vers un tableau : rangées, modules, calibres, noms de circuits.
   Principe : les calibres imprimés sur les disjoncteurs (C10, C16, C20…) servent de repères. Ils sont alignés
   sur chaque rangée et espacés d'un module, ce qui donne les rangées, la largeur d'un module et la place de chacun. */
(function(){
"use strict";
const CAL = [2, 6, 10, 16, 20, 25, 32, 40, 50, 63], AMP = [25, 40, 63, 80, 100, 125];
const CONFUS = {CIO: 10, CI0: 10, C1O: 10, CTO: 10, CT0: 10, CLO: 10, C1G: 16, CIG: 16, CI6: 16, CT6: 16, CL6: 16, C2O: 20, C3Z: 32, CZO: 20, C1B: 16};
const MARQUES = ["SCHNEIDER", "ELECTRIC", "LEGRAND", "HAGER", "MERLINGERIN", "MERLIN", "GERIN", "RESI9", "MULTI9", "DECLIC", "EATON", "SIEMENS", "GEWISS", "LEXMAN", "DEBFLEX", "ACTI9"];
const med = a => { if(!a.length) return 0; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
function lev(a, b){ const n = b.length; let p = Array.from({length: n + 1}, (_, i) => i);
  for(let i = 1; i <= a.length; i++){ const c = [i]; for(let j = 1; j <= n; j++) c[j] = Math.min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)); p = c; } return p[n]; }
// Le mot ressemble-t-il à une marque (ou à un morceau de marque, le logo étant souvent coupé) ?
function marque(N){ if(/^[EF]AT[.·\-]?[O0]?N$/.test(N)) return true; if(N.length < 4) return /^(ABB|AEG|XE|KE|ID|DM|OM)$/.test(N); const tol = N.length >= 7 ? 2 : 1;
  // un logo est souvent coupé d'un côté : on compare au début ou à la fin du nom de la marque
  for(const B of MARQUES) for(let l = Math.max(4, N.length - 1); l <= Math.min(B.length, N.length + 1); l++) if(lev(N, B.slice(0, l)) <= tol || lev(N, B.slice(-l)) <= tol) return true;
  return false; }
// Classe un texte : calibre, différentiel, intensité, levier, marque…
function classe(t){
  const N = t.toUpperCase().replace(/\s+/g, ""), o = {};
  let m = /^(?:DM|OM|0M|DT40|DT4O)?[.\-]?([BCD])(\d{1,3})(.*)$/.exec(N);
  if(m){ const d = m[2], reste = m[3]; let v = 0;
    if(d.length >= 2 && CAL.includes(+d.slice(0, 2))) v = +d.slice(0, 2); else if(d.length === 1 && CAL.includes(+d)) v = +d;
    if(v && (reste.length <= 2 || /^[I1L]?[△ΔA]N|^[^A-Z0-9]/.test(reste))){ o.cal = v; o.courbe = m[1]; } }
  if(!o.cal && CONFUS[N]){ o.cal = CONFUS[N]; o.courbe = "C"; o.flou = true; }
  if(/(^|[^0-9])(30|300|10|500)MA/.test(N)) o.ma = +/(30|300|10|500)MA/.exec(N)[1] === 300 && !/(^|[^0-9])300MA/.test(N) ? 30 : +/(300|30|10|500)MA/.exec(N)[1];
  else if(/0[.,]0?30?A|[△Δ]N0[.,]03/.test(N)) o.ma = 30;
  m = /^(\d{2,3})A(?![A-Z])|^(\d{2,3})A\d{2,3}MA/.exec(N); if(m && AMP.includes(+(m[1] || m[2]))) o.amp = +(m[1] || m[2]);
  m = /^TY[POQ0]EA?(AC|A|F)$|^TY[POQ0]E(AC|A|F)/.exec(N); if(m) o.type = (m[1] || m[2]).toLowerCase();
  if(/^[IL1T][.\-·:]?O[NMH]I?$|^[0O][.\-·]?OFF$|^[.\-]O[NM]$/.test(N)) o.levier = true;
  if(marque(N)) o.marque = true;
  if(/^2[3O0][0O]V|^4[0O]{2}V|^[36]0{3}\d?$|^4500$|^10000$|^0{3}\d?$|EN6\d{4}|^R9[A-Z0-9]{3,}|^A9[A-Z0-9]{3,}|VIGI|TESTM/.test(N) || (!/[a-z]/.test(t) && /^[A-Z]{2,5}C[0-9O]{3}$|^R9[A-Z0-9]+$/.test(N))) o.divers = true;
  o.app = !!(o.cal || o.ma || o.amp || o.type || o.levier || o.marque || o.divers);
  return o;
}
/* Petites corrections de français : le lecteur ne connaît pas bien les accents */
const ACCENTS = {sejour: "séjour", entree: "entrée", differentiel: "différentiel", lumiere: "lumière", lumieres: "lumières", eclairage: "éclairage", seche: "sèche", congelateur: "congélateur",
  refrigerateur: "réfrigérateur", general: "général", generale: "générale", reserve: "réserve", camera: "caméra", cameras: "caméras", exterieur: "extérieur", exterieure: "extérieure", interieur: "intérieur",
  etage: "étage", degagement: "dégagement", telecommande: "télécommande", securite: "sécurité", chaudiere: "chaudière", telerupteur: "télérupteur", piece: "pièce", pieces: "pièces",
  cuisiniere: "cuisinière", plafonnier: "plafonnier", electrique: "électrique", arriere: "arrière", bibliotheque: "bibliothèque", salle: "salle", ete: "été", television: "télévision",
  tele: "télé", telephone: "téléphone", reseau: "réseau", detecteur: "détecteur", fumee: "fumée", video: "vidéo", debarras: "débarras", veranda: "véranda", portail: "portail", sirene: "sirène",
  delesteur: "délesteur", gache: "gâche", facade: "façade", fenetre: "fenêtre", fenetres: "fenêtres", hote: "hôte", chauffe: "chauffe", vehicule: "véhicule", piscine: "piscine", lave: "lave"};
function accent(mot){ const k = mot.toLowerCase(), v = ACCENTS[k]; if(!v || v === k) return mot;
  if(mot === mot.toUpperCase() && mot.length > 1) return v.toUpperCase(); return mot[0] === mot[0].toUpperCase() ? v[0].toUpperCase() + v.slice(1) : v; }
function nomPropre(mots){
  let s = mots.join(" ").replace(/([A-Za-zÀ-ÿ])- (?=[a-zà-ÿ])/g, "$1-").replace(/\s+/g, " ").trim();
  s = s.replace(/[A-Za-zÀ-ÿ]+/g, accent).replace(/\bPC(?=[A-Z][a-zà-ÿ]|[A-Z]{3})/g, "PC ").replace(/^[\s.,;:|\-_]+|[\s,;:|\-_]+$/g, "");
  return s.slice(0, 80); }

function tableau(bs, W, H){
  const T = [];
  for(const b of bs){ const t = String(b.t || "").replace(/[^\x20-\x7EÀ-ÿ△Δ]/g, "").replace(/\s+/g, " ").trim(); if(!t) continue;
    const haut = Math.abs(b.a) > Math.PI / 4, eh = haut ? b.w : b.h;
    // Deux ou trois calibres lus d'un seul tenant (« C20C20 ») : autant de repères, répartis sur la largeur
    const coll = !haut && /^(?:[BCD]\d{2}){2,3}$/.test(t.replace(/\s/g, "").toUpperCase()) ? t.replace(/\s/g, "").toUpperCase().match(/[BCD]\d{2}/g) : null;
    if(coll){ coll.forEach((c, i) => T.push(Object.assign({}, b, {t: c, k: classe(c), haut, eh, w: b.w / coll.length, cx: b.cx + (i - (coll.length - 1) / 2) * b.w / coll.length * Math.cos(b.a), cy: b.cy + (i - (coll.length - 1) / 2) * b.w / coll.length * Math.sin(b.a)}))); continue; }
    T.push(Object.assign({}, b, {t, k: classe(t), haut, eh})); }
  const vide = note => ({larg: 13, rangees: [], note});
  // Regroupe des textes alignés à la même hauteur et proches les uns des autres (une ligne peut pencher un peu)
  const aligne = toks => { const par = toks.map((_, i) => i), trouve = i => { while(par[i] !== i) i = par[i] = par[par[i]]; return i; };
    for(let i = 0; i < toks.length; i++) for(let j = i + 1; j < toks.length; j++){ const a = toks[i], b = toks[j], hm = Math.max(a.eh, b.eh), dx = Math.abs(a.cx - b.cx), dy = Math.abs(a.cy - b.cy);
      if(dx <= 16 * hm && dy <= .55 * hm + .12 * dx) par[trouve(i)] = trouve(j); }
    const gr = {}; toks.forEach((x, i) => { (gr[trouve(i)] = gr[trouve(i)] || []).push(x); });
    return Object.values(gr).map(g => { g.sort((a, b) => a.cx - b.cx); const ec = [];
      for(let i = 1; i < g.length; i++){ const d = g[i].cx - g[i - 1].cx; if(d > 1.2 * Math.max(g[i].eh, g[i - 1].eh)) ec.push(d); }
      let pas = 0; if(ec.length >= 2){ const mn = Math.min(...ec); pas = med(ec.filter(d => d < 1.6 * mn)); }
      return {g, y: med(g.map(x => x.cy)), x0: g[0].cx, x1: g[g.length - 1].cx, pas, ec, h: med(g.map(x => x.eh))}; }); };
  // 1) Repères : calibres et intensités (sûrs), puis marques et leviers (moins sûrs, mais présents sur chaque appareil)
  const durs = T.filter(x => (x.k.cal || x.k.amp) && x.c >= (x.k.flou ? .75 : .6) && !x.haut);
  const LD = aligne(durs); LD.forEach(l => l.dur = true);
  const LL = aligne(T.filter(x => x.k.levier && !x.k.cal && !x.k.amp && x.c >= .6 && !x.haut)).filter(l => l.g.length >= 3);
  const LM = aligne(T.filter(x => x.k.marque && !x.k.levier && !x.k.cal && !x.k.amp && x.c >= .7 && !x.haut && x.t.length >= 5)).filter(l => l.g.length >= 3);
  const lignes = LD.concat(LL, LM);
  if(!lignes.length) return vide("Je n'ai trouvé aucun calibre lisible (C10, C16, C20…) sur la photo. Rapproche-toi du tableau, bien en face, avec de la lumière.");
  // 2) Largeur d'un module sur la photo : l'écart le plus courant entre deux calibres voisins
  const surs = l => l.filter(x => x.pas && x.g.length >= 3).map(x => x.pas);
  let P = med(surs(LD)) || med(surs(LL)) || med(LM.map(l => med(l.ec)).filter(Boolean));
  if(!P){ const deux = LD.filter(l => l.ec.length).map(l => Math.min(...l.ec)); P = deux.length ? med(deux) : 3.4 * med(LD.map(l => l.h)); }
  lignes.forEach(l => { if(!l.dur || !l.pas || l.g.length < 3 || l.pas > 1.6 * P || l.pas < .6 * P) l.pas = P; });
  // 3) Rangées : les lignes proches en hauteur (marque en haut, calibre en bas d'un même disjoncteur) vont ensemble
  lignes.sort((a, b) => a.y - b.y); const rails = [];
  for(const l of lignes){ const r = rails.find(r => l.y - r.yh < 2.9 * Math.max(P, l.pas) && l.x0 < r.x1 + 3 * P && l.x1 > r.x0 - 3 * P);
    if(r){ r.lignes.push(l); r.yb = Math.max(r.yb, l.y); r.x0 = Math.min(r.x0, l.x0); r.x1 = Math.max(r.x1, l.x1); }
    else rails.push({lignes: [l], yh: l.y, yb: l.y, x0: l.x0, x1: l.x1}); }
  for(const r of rails){ const ps = r.lignes.filter(l => l.dur && l.g.length >= 3).map(l => l.pas); r.P = ps.length ? med(ps) : P;
    r.rep = [].concat(...r.lignes.filter(l => l.dur).map(l => l.g)); r.nb = r.lignes.reduce((s, l) => s + l.g.length, 0);
    // Pv : hauteur de référence. De gros appareils (3 ou 4 modules) sont plus espacés : on se fie alors à la taille du texte.
    r.Pv = r.rep.length ? Math.min(r.P, 5.5 * med(r.rep.map(x => x.eh))) : r.P; }
  // Une rangée faite d'un seul repère peu sûr est écartée
  let bons = rails.filter(r => r.nb >= 2 || r.rep.some(x => x.c >= .9 && !x.k.flou));
  if(!bons.length) bons = rails;
  // Nombres seuls (« 20 », « 16 ») posés sur une ligne de calibres : le « C » n'a pas été lu
  for(const r of bons) for(const x of T){ if(x.k.app || x.haut || x.c < .9 || !/^\d{1,2}$/.test(x.t) || !CAL.includes(+x.t)) continue;
    for(const l of r.lignes){ if(!l.dur || l.g.length < 2) continue; const v = l.g.reduce((m, t) => Math.abs(t.cx - x.cx) < Math.abs(m.cx - x.cx) ? t : m);
      if(Math.abs(v.cx - x.cx) < 4 * r.P && Math.abs(v.cx - x.cx) > .6 * r.P && Math.abs(v.cy - x.cy) < .5 * v.eh + .12 * Math.abs(v.cx - x.cx) * .5){ x.k = {cal: +x.t, flou: true, app: true}; r.rep.push(x); r.x0 = Math.min(r.x0, x.cx); r.x1 = Math.max(r.x1, x.cx); break; } } }
  // 4) Bande occupée par les appareils de chaque rangée (tout texte dedans est du texte d'appareil, pas une étiquette)
  for(const r of bons){ const app = T.filter(x => x.k.app && x.cx > r.x0 - 4 * r.P && x.cx < r.x1 + 4 * r.P && x.cy > r.yb - 2.9 * r.Pv && x.cy < r.yh + 2.9 * r.Pv);
    r.app = app; let h = Math.min(r.yh, ...app.map(x => x.cy)), b = Math.max(r.yb, ...app.map(x => x.cy));
    // une seule ligne connue : le calibre est en général imprimé en bas de l'appareil, la marque en haut
    if(b - h < 1.2 * r.Pv){ if(r.rep.length) h = Math.min(h, b - 2.3 * r.Pv); else b = Math.max(b, h + 2.3 * r.Pv); }
    r.bh = Math.max(h - .45 * r.Pv, r.yb - 2.9 * r.Pv); r.bb = Math.min(b + .45 * r.Pv, r.yh + 2.9 * r.Pv); }
  // Deux bandes ne se chevauchent pas : on coupe au milieu
  bons.sort((a, b) => a.yh - b.yh);
  for(let i = 1; i < bons.length; i++){ const a = bons[i - 1], b = bons[i]; if(a.bb > b.bh && Math.min(a.x1, b.x1) > Math.max(a.x0, b.x0) - 2 * P){ const m = (a.yb + b.yh) / 2; a.bb = Math.min(a.bb, m); b.bh = Math.max(b.bh, m); a.app = a.app.filter(x => x.cy < m); b.app = b.app.filter(x => x.cy >= m); } }
  let nonLus = 0, diffs = 0;
  // 6) Appareils de chaque rangée
  for(const r of bons){ const Pr = r.P, cols = [];
    // colonnes de calibre
    for(const x of r.rep.filter(x => x.k.cal).sort((a, b) => a.cx - b.cx)){ const c = cols.find(c => Math.abs(c.x - x.cx) < .42 * Pr);
      if(c){ c.toks.push(x); c.x = c.toks.reduce((s, t) => s + t.cx, 0) / c.toks.length; } else cols.push({x: x.cx, toks: [x]}); }
    const apps = cols.map(c => { const voix = {}; c.toks.forEach(t => { voix[t.k.cal] = (voix[t.k.cal] || 0) + t.c * t.eh * (t.k.flou ? .5 : 1); });
      const cal = +Object.keys(voix).sort((a, b) => voix[b] - voix[a])[0]; return {x: c.x, cal, w: 1, y: med(c.toks.map(t => t.cy))}; });
    // intensités seules (40A, 63A) : interrupteurs différentiels le plus souvent
    for(const x of r.rep.filter(x => x.k.amp && !x.k.cal)){ if(apps.some(a => a.amp && Math.abs(a.x - x.cx) < 1.2 * Pr)) continue;
      const proche = apps.find(a => a.cal && Math.abs(a.x - x.cx) < .6 * Pr); if(proche) continue;
      apps.push({x: x.cx, amp: x.k.amp, w: 2, ma: x.k.ma || 0, y: x.cy}); }
    // marques « 30 mA », « type A »… rattachées à l'appareil le plus proche
    for(const x of r.app){ if(!x.k.ma && !x.k.type) continue; let best = null, bd = 1e9;
      for(const a of apps){ const d = Math.abs(a.x - x.cx) / Pr - (a.amp ? .6 : 0); if(d < bd){ bd = d; best = a; } }
      if(best && bd < 1.3){ if(x.k.ma && !best.ma) best.ma = x.k.ma; if(x.k.type) best.type = x.k.type; if(best.cal && x.k.ma) best.w = 2; } }
    // appareils sans calibre lisible, repérés par leur marque ou leur levier
    const mous = [];
    for(const x of r.app.filter(x => (x.k.marque || x.k.levier) && !x.k.cal && !x.k.amp && x.c >= .6).sort((a, b) => a.cx - b.cx)){
      const c = mous.find(c => Math.abs(c.x - x.cx) < .42 * Pr); if(c){ c.n++; c.x = (c.x * (c.n - 1) + x.cx) / c.n; } else mous.push({x: x.cx, n: 1, y: x.cy}); }
    for(const c of mous){ if(apps.some(a => Math.abs(a.x - c.x) < (a.w / 2 + .38) * Pr)) continue; apps.push({x: c.x, w: 1, inconnu: true, y: c.y}); }
    apps.sort((a, b) => a.x - b.x);
    // deux inconnus collés (logo et levier d'un même gros appareil) : un seul
    for(let i = apps.length - 1; i > 0; i--) if(apps[i].inconnu && apps[i - 1].inconnu && apps[i].x - apps[i - 1].x < .8 * Pr) apps.splice(i, 1);
    r.apps = apps;
  }
  // 7) Étiquettes : texte hors des bandes d'appareils, rattaché à la rangée la plus proche puis à l'appareil en face
  const etiq = T.filter(x => !x.k.app && x.c >= .65 && /[A-Za-zÀ-ÿ]{2}/.test(x.t) && !bons.some(r => x.cy > r.bh && x.cy < r.bb && x.cx > r.x0 - 3 * r.P && x.cx < r.x1 + 3 * r.P));
  let poses = [];
  for(const x of etiq){ let best = null, bd = 1e9, dessus = false;
    for(const r of bons){ if(x.cx < r.x0 - 2.5 * r.P || x.cx > r.x1 + 2.5 * r.P) continue; const d = x.cy < r.bh ? r.bh - x.cy : x.cy - r.bb; if(d < bd){ bd = d; best = r; dessus = x.cy < r.bh; } }
    if(!best || bd > 2.7 * best.Pv) continue;
    let app = null, ad = 1e9; for(const a of best.apps){ const d = Math.abs(a.x - x.cx) - (a.w - 1) * best.P / 2; if(d < ad){ ad = d; app = a; } }
    if(!app || ad > .62 * best.P) continue;
    poses.push({x, app, dessus, d: bd, r: best}); }
  // Sur un tableau, les étiquettes sont du même côté pour toutes les rangées (au-dessus ou en dessous) : le côté minoritaire est écarté
  const nets = poses.filter(e => e.x.c >= .8), nd = nets.filter(e => e.dessus).length, ns = nets.length - nd;
  const cote = nets.length >= 4 && Math.max(nd, ns) >= .6 * nets.length ? (nd > ns ? "dessus" : "dessous") : "";
  if(cote) poses = poses.filter(e => e.dessus === (cote === "dessus"));
  // Pour chaque appareil : on part du texte le plus proche et on remonte tant que les lignes se suivent.
  // Un texte bien lu ouvre la série ; un texte moins net n'est gardé que s'il touche une ligne déjà gardée.
  const parApp = new Map(); poses.forEach(e => { if(!parApp.has(e.app)) parApp.set(e.app, []); parApp.get(e.app).push(e); });
  for(const [app, l] of parApp){ l.sort((a, b) => a.d - b.d); const garde = []; let dern = null;
    for(const e of l){ if(dern !== null && e.d - dern > 1.25 * e.r.Pv) break; garde.push(e); dern = e.d; }
    const ok = garde.filter(e => e.x.c >= .8); if(!ok.length) continue;
    app.etq = garde.filter(e => e.x.c >= .8 || ok.some(o => Math.abs(o.d - e.d) < 1.0 * e.r.Pv)).map(e => e.x); }
  // 8) Mise en forme
  const rangees = [];
  for(const r of bons){ if(!r.apps.length) continue; const mods = []; let pos = 0, prec = null;
    for(const a of r.apps){ const m = {};
      if(a.inconnu){ m.t = "autre"; m.lib = "?"; m.w = 1; nonLus++; }
      else if(a.amp){ diffs++; const ty = a.type || "ac", k = "id" + a.amp + ty; if(a.ma === 300 || ![40, 63].includes(a.amp) || !["ac", "a", "f"].includes(ty) || (ty === "f" && a.amp !== 40)){ m.t = "autre"; m.lib = "ID " + a.amp + "A" + (a.ma ? " " + a.ma + "mA" : ""); m.w = 2; } else m.t = k; }
      else if(a.ma){ if(a.ma === 30 && (a.cal === 16 || a.cal === 20)) m.t = "dd" + a.cal; else { m.t = "autre"; m.lib = "DD " + a.cal + "A " + a.ma + "mA"; m.w = 2; } }
      else if([2, 10, 16, 20, 25, 32, 40].includes(a.cal)) m.t = "d" + a.cal;
      else { m.t = "autre"; m.lib = "Disj. " + a.cal + "A"; m.w = 1; }
      const w = m.w || a.w;
      if(prec){ const trou = Math.round((a.x - prec.x) / r.P - (w + prec.w) / 2); pos += prec.w + Math.max(0, Math.min(12, trou)); }
      m.p = pos; prec = {x: a.x, w};
      if(a.etq){ const mots = a.etq.sort((u, v) => Math.abs(u.cy - v.cy) > .5 * Math.min(u.eh, v.eh) ? u.cy - v.cy : u.cx - v.cx).map(u => u.t);
        // repères seuls (Q7, A3) et rappel du type (« Interrupteur différentiel ») : pas des noms de circuit
        const redite = t => { const n = t.toLowerCase().replace(/[^a-z]/g, ""); return /^diff?$/.test(n) || (n.length > 12 && lev(n, "interrupteurdifferentiel") <= 8); };
        const utiles = mots.filter(t => !/^[A-Z]{1,2}\d{1,2}$/.test(t.replace(/\s/g, "")) && !((a.amp || a.ma) && redite(t))); m.nom = nomPropre(utiles); } else m.nom = "";
      // zone de la photo où se trouve l'appareil (pour l'afficher pendant la vérification)
      const lim = (v, a, b) => Math.max(a, Math.min(b, v)), x0 = lim(a.x - (w / 2 + 1.1) * r.P, 0, W), x1 = lim(a.x + (w / 2 + 1.1) * r.P, 0, W), y0 = lim(r.bh - 1.7 * r.Pv, 0, H), y1 = lim(r.bb + 1.7 * r.Pv, 0, H);
      const d0 = lim(a.x - w / 2 * r.P, x0, x1), d1 = lim(a.x + w / 2 * r.P, x0, x1);
      if(x1 - x0 > 4 && y1 - y0 > 4) m.z = [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H, d0 / W, (d1 - d0) / W].map(v => Math.round(v * 1e4) / 1e4);
      mods.push(m); }
    rangees.push({mods, _fin: pos + prec.w}); }
  const fin = Math.max(0, ...rangees.map(r => r._fin)); rangees.forEach(r => delete r._fin);
  const nb = rangees.reduce((s, r) => s + r.mods.length, 0), noms = rangees.reduce((s, r) => s + r.mods.filter(m => m.nom).length, 0), notes = [];
  if(nonLus) notes.push(nonLus + " module" + (nonLus > 1 ? "s" : "") + " marqué" + (nonLus > 1 ? "s" : "") + " « ? » : je les ai vus mais pas pu les identifier.");
  if(diffs) notes.push("Type des différentiels (AC, A, F) à vérifier.");
  if(noms < nb) notes.push(noms ? "Noms de circuits lus : " + noms + " sur " + nb + "." : "Aucun nom de circuit lu (écriture à la main ou étiquettes trop petites).");
  return {larg: [13, 18, 24].find(n => n >= fin) || 24, rangees, note: notes.join(" ")};
}
const api = {tableau, classe, nomPropre};
if(typeof window !== "undefined"){ window.TBOCR = Object.assign(window.TBOCR || {}, api); }
if(typeof module !== "undefined") module.exports = api;
})();
