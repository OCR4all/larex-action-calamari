set -a
source .env
source .env_eval
set +a
#uv run 
uvicorn eval:app --host 0.0.0.0 --port 19102
