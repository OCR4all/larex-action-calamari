set -a
source .env
set +a
#uv run 
uvicorn predict:app --host 0.0.0.0 --port 19002
