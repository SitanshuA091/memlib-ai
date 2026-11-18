# Merging this into memlib-ai

1. Copy into the repo root (same level as `src/`, `tests/`, `pyproject.toml`):
   - `docs/`
   - `mkdocs.yml`
   - `docs-requirements.txt`
   - `.github/workflows/docs.yml` (into your existing `.github/workflows/`, alongside `release.yml`)

2. Preview locally:
   ```bash
   pip install -r docs-requirements.txt
   mkdocs serve
   ```
   Open http://127.0.0.1:8000

3. Enable Pages (one-time, in the GitHub repo settings):
   - Settings → Pages → Build and deployment → Source: **GitHub Actions**

4. Push to `main`. The `docs.yml` workflow builds and deploys automatically on
   any change under `docs/` or `mkdocs.yml`. Your `release.yml` (PyPI publish
   on release) is untouched and runs independently.

5. Site will be live at:
   ```
   https://SitanshuA091.github.io/memlib-ai/
   ```

6. Optional: add a docs badge/link to `README.md`:
   ```markdown
   [![Docs](https://img.shields.io/badge/docs-live-blue)](https://SitanshuA091.github.io/memlib-ai/)
   ```
