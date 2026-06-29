# Deploying the Toaster web demo

The web server is a plain FastAPI app, packaged as a small Docker image
(`Dockerfile` at the repo root — no PySide6/pywebview, ~600 MB). The **same
image** runs on any container host, so you are not locked into one provider.

```bash
docker build -t toaster-demo .
docker run -p 7860:7860 toaster-demo   # http://127.0.0.1:7860
```

## Hugging Face Spaces (simplest, free)

1. Create a new Space → **SDK: Docker** (blank).
2. Give its README this frontmatter (top of the Space's `README.md`):

   ```yaml
   ---
   title: Toaster Demo
   emoji: 🍞
   colorFrom: red
   colorTo: gray
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

   (Or skip it and set the app port to **7860** in the Space settings.)
3. Push this repo to the Space's git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/toaster-demo
   git push space HEAD:main
   ```

HF builds the `Dockerfile` and serves the app. Free Spaces sleep after
inactivity and wake on the next visit (~30 s).

## Other hosts (same Dockerfile)

- **Render** — New → Web Service → from repo → "Docker"; it auto-detects the
  Dockerfile. Set the port to 7860 (or rely on `$PORT`).
- **Fly.io** — `fly launch` (detects the Dockerfile), then `fly deploy`.
- **Google Cloud Run** — `gcloud run deploy toaster-demo --source . --port 7860`.
- **Any VPS** — `docker run -p 80:7860 toaster-demo`.

The server reads `$PORT` (defaults to 7860) and opens
`examples/sample.bin` on startup.
