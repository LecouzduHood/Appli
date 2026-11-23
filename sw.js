const CACHE = "devis-v16";
const OCR = "lecteur-v1";   // lecteur de photos (30 Mo) : gardé à part pour ne pas le retélécharger à chaque mise à jour
const FILES = ["./", "index.html", "manifest.json", "icon.svg", "icon-192.png", "icon-512.png", "apple-touch-icon.png", "jspdf.umd.min.js", "pdf.min.js", "pdf.worker.min.js", "docs/came-bxv-zn7.pdf", "atelier-cablage.html", "tb-ocr.js"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES))); self.skipWaiting(); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE && k !== OCR).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin) return;   // jamais les requêtes Google / externes
  // Lecteur de photos : fichiers lourds qui ne changent pas → cache d'abord, réseau la première fois
  if (u.pathname.includes("/ocr/")) {
    e.respondWith(caches.open(OCR).then(c => c.match(e.request).then(r => r || fetch(e.request).then(n => { if (n.ok) c.put(e.request, n.clone()); return n; }))));
    return;
  }
  // Réseau d'abord (pour recevoir les mises à jour), cache si hors ligne
  e.respondWith(fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; })
    .catch(() => caches.match(e.request).then(r => r || caches.match("index.html"))));
});
