## Docker deployment

Build and start the application with:

```powershell
docker compose up -d --build
```

The public chat interface is available at `http://localhost:5008/chat`.
The internal AI Control Center is at `http://localhost:5008/admin`.
The API health check is at `http://localhost:5008/`.

Set these values in `.env`; Compose passes them to the API but `.dockerignore` prevents
`.env` from being copied into the image:

```dotenv
GROQ_API_KEY=your-secret-key
ADMIN_USERNAME=vmschat
ADMIN_PASSWORD=change-this-before-production
ADMIN_COOKIE_SECURE=false
```

Use `ADMIN_COOKIE_SECURE=true` when the application is served behind HTTPS. Admin login
sessions expire five minutes after login. The `data/` bind mount (`./data:/app/data:rw`) persists:
- The active PDF in `data/kb/` and manifest in `data/kb_manifest.json`
- The vector store in `data/chroma_db/`
- The BM25 index cache in `data/cache/bm25_index.pkl`
- The admin configuration in `data/admin_config.json`
- Application logs in `data/logs/chat.log`

The API container intentionally runs one Uvicorn worker so runtime Control Center changes,
knowledge base rebuild staging locks, in-memory chat sessions, and the in-memory response cache stay consistent.
