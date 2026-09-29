# Lean server image for the online demo (no PySide6 / pywebview / pyvista —
# the web server never imports them). Runs the same FastAPI app + web UI.
# Portable: the same image runs on Hugging Face Spaces, Render, Fly.io, Cloud Run…
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Only the server's runtime deps, then the package itself without its (desktop) deps.
# projector-engine (the shared 3D engine) is not on PyPI yet: fetched from its
# GitHub release tag as an archive, since the slim image ships no git.
RUN pip install --no-cache-dir \
      numpy scipy "scikit-learn>=1.3" laspy plyfile pyyaml fastapi "uvicorn[standard]" \
      "projector-engine @ https://github.com/augustin-bresset/projector/archive/refs/tags/v0.1.0.tar.gz" \
 && pip install --no-cache-dir --no-deps . \
 && python examples/make_sample.py

# Hugging Face Spaces sets PORT=7860; default for local runs too.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "toaster-web examples/sample.bin --host 0.0.0.0 --port ${PORT}"]
