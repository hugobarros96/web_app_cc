#!/usr/bin/env bash
#
# One-command deploy, run from your laptop at the repo root.
#
#   ./deploy.sh         # GitHub push + portfolio→VM (with artifacts) + Data Doctor→HF Space
#   ./deploy.sh vm      # only: push to GitHub, rsync artifacts, redeploy the portfolio on the VM
#   ./deploy.sh hf      # only: deploy Data Doctor (code + artifacts) to the HF Space
#   ./deploy.sh github  # only: push the current branch to GitHub
#
# What goes where:
#   - The portfolio (this repo) deploys to the GCP VM: the gitignored artifacts/
#     (videos, profile photo, CV, summary, …) is rsync'd up first, then a git
#     pull + docker compose up. The VM runs only the light `web` image.
#   - Data Doctor deploys to its HF Space: everything under
#     projects/health_assistant/ (code, data, and the gitignored artifacts/ with
#     the FAISS indices + models), pushed via git-LFS. Self-contained to THIS
#     repo (no external source repo). Unchanged LFS artifacts are not re-uploaded.
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
PROJECT_SUBDIR="projects/health_assistant"  # self-contained: code, data, and artifacts/ all live here
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
# Disable the pass/GPG credential helper for remote git ops; we authenticate via
# tokens embedded in the URLs, and the helper otherwise hangs on gpg decryption.
GIT="git -c credential.helper="

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
  $GIT push "https://${PAT}@${GH_REPO}" "$BRANCH"
  echo "✓ GitHub up to date"
}

deploy_vm() {
  echo "▶ Syncing artifacts/ to the VM…"
  # artifacts/ is gitignored, so git pull never carries it — rsync it directly.
  # README.md is tracked in git (arrives via git pull), so exclude it here.
  rsync -az --exclude README.md ./artifacts/ "${VM_USER}@${VM_HOST}:${VM_DIR}/artifacts/"
  echo "▶ Redeploying portfolio on the VM ($VM_HOST)…"
  # --build is REQUIRED in prod: it has no source bind-mount, so the container
  # runs whatever is baked into the image. Without it, compose reuses the stale
  # image and code changes never ship.
  ssh "${VM_USER}@${VM_HOST}" "cd ${VM_DIR} && git pull && ${COMPOSE} down && ${COMPOSE} up -d --build && ${COMPOSE} ps"
  echo "✓ VM redeployed"
}

deploy_hf() {
  echo "▶ Deploying Data Doctor to the HF Space ($HF_USER/$HF_SPACE)…"
  [ -n "$HF_TOKEN" ] || { echo "✗ HF_TOKEN not found in .secrets"; exit 1; }
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  # Skip LFS smudge on clone so we DON'T download the ~330 MB of artifacts —
  # they arrive as pointers. We rsync the real files over them below; unchanged
  # artifacts hash to the same pointer (no commit, no re-upload), changed ones
  # get pushed via git-LFS.
  GIT_LFS_SKIP_SMUDGE=1 $GIT clone --quiet \
    "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" "$tmp"
  # Mirror everything under projects/health_assistant/ (code, data, AND the
  # gitignored artifacts/ FAISS indices + models) into the Space clone. If the
  # local artifacts/ is missing/empty, exclude it from the mirror so --delete
  # can't wipe the artifacts already on the Space.
  local extra_excludes=()
  if [ -d "${PROJECT_SUBDIR}/artifacts" ] && [ -n "$(ls -A "${PROJECT_SUBDIR}/artifacts" 2>/dev/null)" ]; then
    echo "  including artifacts/ (FAISS indices + models)"
  else
    echo "  ! ${PROJECT_SUBDIR}/artifacts empty/missing — preserving the Space's existing artifacts"
    extra_excludes=(--exclude 'artifacts/')
  fi
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    "${extra_excludes[@]}" \
    "${PROJECT_SUBDIR}/" "$tmp/"
  (
    cd "$tmp"
    git lfs install --local >/dev/null 2>&1 || true
    git add -A
    if git diff --cached --quiet; then
      echo "✓ HF Space already up to date"
    else
      git commit --quiet -m "auto-deploy: sync Data Doctor"
      $GIT push --quiet
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
