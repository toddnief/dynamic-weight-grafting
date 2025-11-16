#!/bin/bash
#SBATCH --job-name=kg_tmdb_wiki
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Defaults; can be overridden via env when calling make
OUTPUT_PATH=${OUTPUT_PATH:-tmdb_wiki_chunks_with_rewrites.jsonl}
MAX_MOVIES=${MAX_MOVIES:-20}
NUM_REWRITES=${NUM_REWRITES:-0}
MIN_POPULARITY=${MIN_POPULARITY:-10.0}
MAX_CHARS=${MAX_CHARS:-4000}
OVERLAP=${OVERLAP:-200}

echo "[tmdb_wiki] output=${OUTPUT_PATH} max_movies=${MAX_MOVIES} num_rewrites=${NUM_REWRITES}"

python tmdb_wiki_pipeline.py \
  --output "${OUTPUT_PATH}" \
  --max-movies "${MAX_MOVIES}" \
  --min-popularity "${MIN_POPULARITY}" \
  --max-chars "${MAX_CHARS}" \
  --overlap "${OVERLAP}" \
  --num-rewrites "${NUM_REWRITES}"