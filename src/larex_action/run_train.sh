set -a
source .env
source .env_train
set +a
#uv run 
uvicorn train:app --host 0.0.0.0 --port 19202
