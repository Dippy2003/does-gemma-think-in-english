#!/usr/bin/env bash
# Populate app/cache.sqlite3 with every example in app/examples.json, so a
# cold CPU-only Space demos instantly instead of loading the model live.
set -euo pipefail

python3 -c "
import json
from app.cache import get_connection, get_or_compute
from app.trace import trace_prompt
from src.model import load_model

with open('app/examples.json', encoding='utf-8') as f:
    examples = json.load(f)

model = load_model('gemma-2-2b')
conn = get_connection()
for ex in examples:
    get_or_compute(conn, ex['sinhala'], lambda p: trace_prompt(model, p))
    print(f\"cached: {ex['sinhala'][:30]}\")
"
