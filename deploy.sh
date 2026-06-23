#!/usr/bin/env bash
#
# One-command deploy, run from your laptop at the repo root.
#
#   ./deploy.sh         # push to GitHub, deploy the portfolio to the VM,
#                       # then sync Data Doctor code to the HF Space
#   ./deploy.sh vm      # only: push to GitHub + redeploy the portfolio on the VM
#   ./deploy.sh hf      # only: sync Data Doctor code to the HF Space
#   ./deploy.sh github  # only: push the current branch to GitHub
#
# What goes where:
#   - The portfolio (this repo) deploys to the GCP VM the usual way
#     (git pull + docker compose up). The VM runs only the light `web` image.
#   - Data Doctor's CODE (app/, src/, Dockerfile, docs) syncs to the HF Space,
#     which rebuilds and runs the heavy ML container on HF's 16 GB infra.
#     The ~330 MB FAISS/model artifacts already live on the Space (pushed once
#     via DEPLOY_HF.md) and are LEFT UNTOUCHED here — this only updates code.
#
# Secrets are read from .secrets (PAT=, HF_TOKEN=). Never commit them.

set -euo pipefail

# --- config ---------------------------------------------------------------
VM_USER="hugobarros96"
VM_HOST="35.231.149.237"
VM_DIR="~/code/web_app_cc"
GH_REPO="github.com/hugobarros96/web_app_cc.git"
HF_USER="hugobarros96"            # ⚠ your Hugging Face username
HF_SPACE="datadoctor"
PROJECT_SUBDIR="projects/health_assistant"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- secrets --------------------------------------------------------------
secret() { grep -E "^$1=" .secrets | head -1 | cut -d= -f2-; }
PAT="$(secret PAT || true)"
HF_TOKEN="$(secret HF_TOKEN || true)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

deploy_github() {
  echo "▶ Pushing $BRANCH to GitHub…"
  [ -n "$PAT" ] || { echo "✗ PAT not found in .secrets"; exit 1; }
  git push "https://${PAT}@${GH_REPO}" "$BRANCH"
  echo "✓ GitHub up to date"
}

deploy_vm() {
  echo "▶ Redeploying portfolio on the VM ($VM_HOST)…"
  ssh "${VM_USER}@${VM_HOST}" "cd ${VM_DIR} && git pull && ${COMPOSE} down && ${COMPOSE} up -d && ${COMPOSE} ps"
  echo "✓ VM redeployed"
}

deploy_hf() {
  echo "▶ Syncing Data Doctor code to the HF Space ($HF_USER/$HF_SPACE)…"
  [ -n "$HF_TOKEN" ] || { echo "✗ HF_TOKEN not found in .secrets"; exit 1; }
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  # Skip LFS smudge so we DON'T download the 330 MB of artifacts — they come
  # down as pointers and are left exactly as-is on push.
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet \
    "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" "$tmp"
  # Mirror code into the Space clone, but never touch artifacts/ or .git/.
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'artifacts/' \
    --exclude '__pycache__/' \
    "${PROJECT_SUBDIR}/" "$tmp/"
  (
    cd "$tmp"
    git add -A
    if git diff --cached --quiet; then
      echo "✓ HF Space already up to date (no code changes)"
    else
      git commit --quiet -m "auto-deploy: sync Data Doctor code"
      git push --quiet
      echo "✓ HF Space updated — it will rebuild automatically"
    fi
  )
}

case "${1:-all}" in
  github) deploy_github ;;
  vm)     deploy_github; deploy_vm ;;
  hf)     deploy_hf ;;
  all)    deploy_github; deploy_vm; deploy_hf ;;
  *) echo "usage: ./deploy.sh [all|vm|hf|github]"; exit 1 ;;
esac

echo "✅ Done."
