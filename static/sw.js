// Service worker for PWA installability
// This app runs on a local network (LAN), so no offline caching is needed.
// The service worker is required by browsers to enable the "Add to Home Screen" prompt.

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));
