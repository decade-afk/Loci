#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_ROOT="${REPO_ROOT}/android-sdk/loci-sdk/src/main/jniLibs"

copy_if_exists() {
  local abi="$1"
  local triple="$2"
  local source="${REPO_ROOT}/target/${triple}/release/libloci.so"
  if [[ -f "${source}" ]]; then
    mkdir -p "${DEST_ROOT}/${abi}"
    cp "${source}" "${DEST_ROOT}/${abi}/libloci.so"
    printf '%s\n' "${abi}"
  fi
}

copied=()
while IFS= read -r abi; do
  copied+=("${abi}")
done < <(
  copy_if_exists "arm64-v8a" "aarch64-linux-android"
  copy_if_exists "armeabi-v7a" "armv7-linux-androideabi"
  copy_if_exists "x86_64" "x86_64-linux-android"
  copy_if_exists "x86" "i686-linux-android"
)

if [[ "${#copied[@]}" -eq 0 ]]; then
  echo "No Android libloci.so artifacts found under target/<triple>/release. Build them from the repository root first." >&2
  exit 1
fi

printf 'Synced Loci Android native libraries for: %s\n' "${copied[*]}"
