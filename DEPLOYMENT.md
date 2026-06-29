# Deploy RAGGym

This repo is ready to deploy as a Streamlit app from `streamlit_app.py`.

## Recommended Quick Deploy: Streamlit Community Cloud

1. Push this branch to GitHub.
2. Open Streamlit Community Cloud and create a new app from this repo.
3. Set the app entrypoint to:

```text
streamlit_app.py
```

4. Add these secrets in the Streamlit app settings:

```toml
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = "sk-..."

EMBED_PROVIDER = "openai"
EMBED_MODEL = "text-embedding-3-small"

VECTOR_STORE = "qdrant"
QDRANT_COLLECTION = "raggym"

ENABLE_CAPTIONING = true
VISION_PROVIDER = "openai"
VISION_MODEL = "gpt-4o-mini"
```

## Persistence

For a short demo, you can leave `QDRANT_URL` unset. The app can ingest uploaded
PDFs into local Qdrant storage, but that storage belongs to the app container and
may disappear when the host rebuilds or restarts it.

For a real deployed app, use Qdrant Cloud or another persistent Qdrant server and
add these secrets:

```toml
QDRANT_URL = "https://YOUR-CLUSTER-URL"
QDRANT_API_KEY = "YOUR-QDRANT-KEY"
```

Then upload PDFs in the sidebar and click `Save + ingest`. The generated chunks
and visual captions will be stored in the remote Qdrant collection.

## Notes

- Do not commit `.env`; configure secrets in the hosting platform.
- `requirements.txt` installs the package with the `rag`, `ingest`, and `chat`
  extras needed by the deployed UI.
- `runtime.txt` pins Python 3.11 for cloud compatibility.
- Vision captioning uses extra OpenAI calls during ingestion. Turn it off with
  `ENABLE_CAPTIONING=false` if you want faster/cheaper ingestion.
