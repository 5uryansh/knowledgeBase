"""Sample conversations, part 2: hardware / IoT and web development."""

CONVERSATIONS = [
    ("chatgpt", "2025-11-20", "ESP32 e-paper weather station: partial refresh ghosting", [
        "I'm building a weather station with an ESP32 and a 1.54 inch e-paper display using GxEPD2. Partial refresh leaves ghosting. How do I fix it?",
        "Ghosting is expected with repeated partial updates on e-paper. Do a full refresh every N partial updates (for example every 10), and call display.setPartialWindow only for the region that changes. With GxEPD2, use display.firstPage()/nextPage() for paged drawing, and make sure you call display.hibernate() after drawing so the panel is not left powered. Ghosting also gets worse in cold temperatures.",
        "The ESP32 drains my battery in two days.",
        "Use deep sleep between updates: esp_sleep_enable_timer_wakeup for the refresh interval, and keep Wi-Fi off except when fetching the weather. A typical cycle wakes, connects, fetches JSON over HTTPS, draws a partial update, hibernates the display and goes back to deep sleep. That brings average current into the tens of microamps and battery life into months.",
        "How do I fetch the weather data?",
        "Call a weather API with HTTPClient and parse it with ArduinoJson. Or have your own small backend aggregate the data and return a compact JSON tailored to the display, which keeps the firmware simple and the payload small.",
    ]),
    ("claude-code", "2025-12-01", "esp32-weather firmware: add rotary encoder page navigation", [
        "Add rotary encoder support to the ESP32 firmware so I can switch between pages: weather, system monitor, stopwatch.",
        "I added an encoder module using the ESP32 PCNT peripheral for glitch-free counting, with a debounce of 2 ms on the push button. Rotating changes the page index, and each page has a draw function that performs a partial refresh of the e-paper display. The main loop now wakes from light sleep on the encoder GPIO interrupt instead of deep sleep, because deep sleep resets the PCNT counter.",
        "The stopwatch page flickers on every second.",
        "I limited the stopwatch page to a partial window covering just the digits and increased the full-refresh interval to every 60 updates. GxEPD2 partial refresh on this panel takes about 300 ms, so updating once per second is near the limit. I also switched the font to a pre-rendered bitmap font to reduce drawing time.",
    ]),
    ("gemini", "2025-12-20", "MQTT vs HTTP for ESP32 sensors", [
        "Should my ESP32 sensors use MQTT or HTTP to send readings to Home Assistant?",
        "MQTT is usually better: it is lightweight, keeps a persistent connection and Home Assistant has native MQTT discovery, so sensors appear automatically. Run the Mosquitto broker as a Home Assistant add-on or in Docker. HTTP is simpler for battery devices that wake rarely, because MQTT's connection setup costs energy each time. For deep-sleep devices, publish with QoS 1 and retain so the last value is available.",
    ]),
    ("chatgpt", "2026-01-25", "TinyML: running a quantized model on the ESP32", [
        "Can I run a small neural network on an ESP32-S3 for keyword spotting?",
        "Yes. Train a small CNN on MFCC features, then convert it with TensorFlow Lite and apply full int8 quantization; quantization is what makes it fit in the S3's memory and run fast on its vector instructions. Deploy with TensorFlow Lite Micro or ESP-DL. A keyword-spotting model of about 50k parameters runs inference in a few milliseconds on the ESP32-S3.",
        "Is it the same quantization as for LLMs?",
        "The principle is the same, mapping floats to int8 with a scale and zero point, but for microcontrollers you quantize activations too and use integer-only kernels, because there is no fast floating-point path. LLM methods like GPTQ and AWQ focus on 4-bit weights for memory bandwidth; TinyML quantization targets integer arithmetic for compute.",
    ]),
    ("claude-code", "2026-02-08", "esp32-weather backend: FastAPI endpoint for the display", [
        "Create a small FastAPI backend that aggregates weather and my Spotify now-playing info into one compact JSON for the ESP32 display.",
        "Done. The FastAPI app has GET /display returning {weather, spotify, updated_at}, with a 60 second cache so the ESP32 can poll cheaply. The Spotify part uses the refresh-token flow and returns track, artist and a 1-bit dithered album-art bitmap sized for the 200x200 e-paper display. The backend runs in Docker Compose with Caddy as a reverse proxy for HTTPS.",
        "Album art dithering looks muddy.",
        "Switched from simple thresholding to Floyd–Steinberg dithering with Pillow, plus a contrast boost before dithering. On e-paper the error-diffusion dithering keeps much more detail. The bitmap is packed as bytes so the ESP32 can draw it directly with drawBitmap.",
    ]),
    ("chatgpt", "2025-11-08", "Next.js App Router: server components vs client components", [
        "I'm confused about React Server Components in the Next.js App Router. When do I use 'use client'?",
        "Server Components are the default in the App Router: they render on the server, can fetch data directly (even query the database) and send no JavaScript for themselves to the browser. Add 'use client' only for components that need state, effects, event handlers or browser APIs. A common pattern: a server component page fetches data and passes it as props to small interactive client components.",
        "How do I fetch data with Prisma inside a server component?",
        "Import the Prisma client in the server component and await prisma.post.findMany() directly; no API route needed. Instantiate Prisma once (a global singleton in development) to avoid exhausting PostgreSQL connections during hot reload.",
    ]),
    ("claude", "2025-12-10", "Prisma migrations with PostgreSQL in Docker", [
        "Set up Prisma with PostgreSQL running in Docker Compose for local development.",
        "Add a postgres service to docker-compose.yml with a named volume, expose 5432, and set DATABASE_URL=postgresql://app:app@localhost:5432/app in .env. Then npx prisma migrate dev --name init creates the tables and generates the client. In CI, use prisma migrate deploy instead of migrate dev.",
        "Migration fails with 'relation already exists'.",
        "The database already has tables that Prisma's migration history does not know about. Either reset the dev database with prisma migrate reset, or baseline: create the initial migration with prisma migrate diff and mark it applied with prisma migrate resolve --applied. Baselining is the right move for an existing production database.",
    ]),
    ("chatgpt", "2026-01-14", "Zero sync engine vs tRPC for a real-time app", [
        "I'm evaluating Zero (the sync engine from Rocicorp) for a collaborative Next.js app. How does it compare to tRPC?",
        "tRPC gives you type-safe request/response APIs: the client calls procedures and you handle caching with React Query. Zero is a sync engine: it keeps a client-side replica of the data you query and syncs changes in real time from PostgreSQL, so reads are instant and updates propagate to all clients automatically. Zero is an alternative to building real-time features on top of tRPC plus websockets. It works with Postgres logical replication.",
        "How did I implement Zero sync with Prisma then?",
        "Keep Prisma for schema and migrations, and point Zero at the same PostgreSQL database with wal_level=logical. Zero has its own schema definition for the synced tables and permissions; generate it from your Prisma schema so the two stay consistent. Writes go through Zero mutators so they are applied optimistically on the client and confirmed by the server.",
    ]),
    ("claude-code", "2026-03-02", "notes-app: add authentication with NextAuth", [
        "Add GitHub login to the notes app using NextAuth (Auth.js) with the Prisma adapter.",
        "Added Auth.js with the GitHub provider and the Prisma adapter, so users, accounts and sessions are stored in PostgreSQL. The auth() helper is used in server components to read the session, and middleware protects the /notes routes. I added the Account, Session, User and VerificationToken models to schema.prisma and created a migration.",
        "Login works locally but fails on Vercel.",
        "AUTH_SECRET and the GitHub callback URL were missing in production. I documented AUTH_SECRET, AUTH_GITHUB_ID and AUTH_GITHUB_SECRET in .env.example, and the GitHub OAuth app needs the callback https://<domain>/api/auth/callback/github. Also set AUTH_TRUST_HOST=true when running behind a proxy.",
    ]),
    ("gemini", "2026-03-20", "Tailwind CSS dark mode with Next.js", [
        "Best way to add dark mode with Tailwind CSS in Next.js?",
        "Set darkMode: 'class' in the Tailwind config (or use the selector strategy in Tailwind v4) and toggle a dark class on the html element. The next-themes package handles persistence and avoids the flash of the wrong theme by injecting a small script before hydration. Then use dark: variants like dark:bg-zinc-900 in your components.",
    ]),
]
