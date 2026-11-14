// Lottery Predictor Service Worker
// Provides offline capability for the Streamlit PWA

const CACHE_NAME = 'lottery-predictor-v1';
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  'https://cdn.jsdelivr.net/npm/@streamlit/runtime@1.0.0/dist/runtime.js',
  'https://cdn.jsdelivr.net/npm/@streamlit/runtime@1.0.0/dist/runtime.css'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name !== CACHE_NAME)
            .map((name) => caches.delete(name))
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http requests
  if (!event.request.url.startsWith('http')) {
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          // Return cached response and update in background
          event.waitUntil(updateCache(event.request));
          return cachedResponse;
        }

        // Fetch from network
        return fetch(event.request)
          .then((networkResponse) => {
            // Don't cache non-success responses
            if (!networkResponse || networkResponse.status !== 200) {
              return networkResponse;
            }

            // Clone response for cache
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME)
              .then((cache) => cache.put(event.request, responseToCache));

            return networkResponse;
          })
          .catch(() => {
            // Offline fallback for navigation requests
            if (event.request.mode === 'navigate') {
              return caches.match('/');
            }
            // Return a generic offline response
            return new Response(
              JSON.stringify({ error: 'Offline', message: 'Please check your connection' }),
              { status: 503, headers: { 'Content-Type': 'application/json' } }
            );
          });
      })
  );
});

// Background cache update
async function updateCache(request) {
  try {
    const response = await fetch(request);
    if (response && response.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(request, response);
    }
  } catch (error) {
    console.log('[SW] Background update failed:', error);
  }
}

// Push notification handling
self.addEventListener('push', (event) => {
  if (!event.data) return;

  const data = event.data.json();
  const options = {
    body: data.body || 'New lottery draw available!',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    vibrate: [100, 50, 100],
    data: {
      url: data.url || '/'
    },
    actions: [
      { action: 'view', title: 'View Predictions' },
      { action: 'dismiss', title: 'Dismiss' }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Lottery Predictor', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/')
    );
  }
});

// Periodic background sync (for data updates)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'update-lottery-data') {
    event.waitUntil(updateLotteryData());
  }
});

async function updateLotteryData() {
  try {
    // Trigger data update through the app
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({ type: 'UPDATE_DATA' });
    });
  } catch (error) {
    console.log('[SW] Periodic sync failed:', error);
  }
}

// Message handling from main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});