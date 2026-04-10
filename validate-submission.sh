#!/usr/bin/env bash
#
# validate-submission.sh — ExamForge Submission Validator
# Based on official Meta PyTorch OpenEnv Hackathon validator
#
# Usage:
#   ./validate-submission.sh https://igris69-examforge-env.hf.space [repo_dir]
#   ./validate-submission.sh https://igris69-examforge-env.hf.space .
#
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

# Color support
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BOLD='\033[1m'; NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
    local secs="$1"; shift
    if command -v timeout &>/dev/null; then
        timeout "$secs" "$@"
    else
        "$@"
    fi
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+${CLEANUP_FILES[@]}}"; }
trap cleanup EXIT

PING_URL="${1:-https://igris69-examforge-env.hf.space}"
REPO_DIR="${2:-.}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
    printf "Error: directory '%s' not found\n" "${2:-.}"
    exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
    printf "\n"
    printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix above before continuing.\n" "$1"
    exit 1
}

printf "\n"
printf "${BOLD}============================================${NC}\n"
printf "${BOLD}  ExamForge — OpenEnv Submission Validator${NC}\n"
printf "${BOLD}============================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ── Step 1: Ping HF Space ────────────────────────────────────────────────────
log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/reset) ..."
CURL_OUT=$(mktemp /tmp/validate-XXXXXX)
CLEANUP_FILES+=("$CURL_OUT")
HTTP_CODE=$(curl -s -o "$CURL_OUT" -w "%{http_code}" \
    -X POST -H "Content-Type: application/json" -d '{}' \
    "$PING_URL/reset" --max-time 30 2>/dev/null || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
    fail "HF Space not reachable (connection failed or timed out)"
    hint "Try: curl -X POST $PING_URL/reset"
    stop_at "Step 1"
else
    fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
    hint "Check the Space is Running at: $PING_URL"
    stop_at "Step 1"
fi

# ── Step 2: Check inference.py log format ────────────────────────────────────
log "${BOLD}Step 2/4: Checking inference.py log format${NC} ..."
if [ ! -f "$REPO_DIR/inference.py" ]; then
    fail "inference.py not found in repo root"
    stop_at "Step 2"
fi

# Verify format strings exist
if grep -q "\[START\].*task=.*env=.*model=" "$REPO_DIR/inference.py" && \
   grep -q "\[STEP\].*step=.*action=.*reward=.*done=.*error=" "$REPO_DIR/inference.py" && \
   grep -q "\[END\].*success=.*steps=.*score=.*rewards=" "$REPO_DIR/inference.py"; then
    pass "inference.py contains correct [START]/[STEP]/[END] format strings"
else
    fail "inference.py is missing required log format fields"
    hint "Required: [START] task=X env=X model=X"
    hint "Required: [STEP] step=N action=X reward=0.00 done=bool error=null"
    hint "Required: [END] success=bool steps=N score=0.000 rewards=r1,r2"
    stop_at "Step 2"
fi

# ── Step 3: Docker build ─────────────────────────────────────────────────────
log "${BOLD}Step 3/4: Running docker build${NC} ..."
if ! command -v docker &>/dev/null; then
    fail "docker not found"
    hint "Install Docker: https://docs.docker.com/get-docker/"
    stop_at "Step 3"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR/server"
else
    fail "No Dockerfile found"
    stop_at "Step 3"
fi

log "  Building from: $DOCKER_CONTEXT"
BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
    docker build -t examforge-validate:test "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
    pass "Docker build succeeded"
    docker rmi examforge-validate:test &>/dev/null || true
else
    fail "Docker build failed"
    printf "%s\n" "$BUILD_OUTPUT" | tail -20
    stop_at "Step 3"
fi

# ── Step 4: openenv validate ─────────────────────────────────────────────────
log "${BOLD}Step 4/4: Running openenv validate${NC} ..."
if ! command -v openenv &>/dev/null; then
    fail "openenv CLI not found"
    hint "pip install openenv-core"
    stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
    pass "openenv validate passed"
    [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
    fail "openenv validate failed"
    printf "%s\n" "$VALIDATE_OUTPUT"
    stop_at "Step 4"
fi

printf "\n"
printf "${BOLD}============================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  ExamForge is ready to submit.${NC}\n"
printf "${BOLD}============================================${NC}\n"
printf "\n"
exit 0
