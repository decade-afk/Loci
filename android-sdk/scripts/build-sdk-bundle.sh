#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v gradle >/dev/null 2>&1; then
  echo "gradle was not found on PATH. Install Gradle or build from Android Studio." >&2
  exit 1
fi

"${SCRIPT_DIR}/sync-prebuilt-loci.sh"
gradle --no-daemon --stacktrace -p "${REPO_ROOT}/android-sdk" :loci-sdk:assembleRelease :sample-app:assembleDebug
