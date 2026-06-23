#!/usr/bin/env bash
# Keep the Data Doctor Hugging Face Space awake.
#
# HF free Spaces sleep after 48h idle; the next visitor then waits ~30s+ for a
# cold start. A daily ping resets the inactivity timer so the Space never
# sleeps, which also keeps its ML models warm so first loads stay fast.
#
# Installed as a VM cron job (see scripts/install-keepalive-cron.sh comment or
# the CLAUDE.md deploy notes). Pings the Space root; logs one line per run.
set -u
URL="${DATADOCTOR_URL:-https://hugobarros96-datadoctor.hf.space}"
ts() { date -u +%FT%TZ; }
if curl -fsS -m 90 -o /dev/null "$URL/"; then
  echo "$(ts) keepalive ok: $URL"
else
  echo "$(ts) keepalive FAILED: $URL" >&2
fi
