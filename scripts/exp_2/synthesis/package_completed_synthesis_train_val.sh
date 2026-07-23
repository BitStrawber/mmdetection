#!/usr/bin/env bash
set -euo pipefail

# Package completed synthesis train/val splits into one uncompressed tar per
# method. Archives contain the standard synthetic_imagenet directory layout.
#
# Typical first run after an interrupted archive:
#   RESET_PARTIAL=1 \
#   bash scripts/exp_2/synthesis/package_completed_synthesis_train_val.sh
#
# Resume after a later interruption:
#   RESET_PARTIAL=1 \
#   bash scripts/exp_2/synthesis/package_completed_synthesis_train_val.sh
#
# Existing final .tar files are skipped. Only stale .tar.partial files are
# removed when RESET_PARTIAL=1, after checking that no process holds them.

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/media/HDD2/XCX/exp_2/transfer_archives}"
SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
UWDF_ROOT="${UWDF_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_controlnet_ipadapter}"
METHODS="${METHODS:-uwnr syreanet uwdf}"
RESET_PARTIAL="${RESET_PARTIAL:-0}"
REPLACE_ARCHIVE="${REPLACE_ARCHIVE:-0}"
VERIFY_ARCHIVE="${VERIFY_ARCHIVE:-0}"
TRAIN_EXPECTED="${TRAIN_EXPECTED:-250000}"
VAL_EXPECTED="${VAL_EXPECTED:-10000}"

mkdir -p "${ARCHIVE_ROOT}"

count_images() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi

  find "${path}" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
       -o -iname '*.bmp' -o -iname '*.webp' \) \
    2>/dev/null | wc -l
}

human_bytes() {
  numfmt --to=iec-i --suffix=B "$1"
}

partial_is_held() {
  local partial="$1"

  if command -v fuser >/dev/null 2>&1; then
    if fuser "${partial}" >/dev/null 2>&1; then
      return 0
    fi
    return 1
  fi

  if command -v lsof >/dev/null 2>&1; then
    if lsof -t -- "${partial}" 2>/dev/null | grep -q .; then
      return 0
    fi
    return 1
  fi

  echo "Error: neither fuser nor lsof is available to check the partial archive." >&2
  echo "Refusing to remove it without an open-file check: ${partial}" >&2
  exit 1
}

prepare_partial_path() {
  local partial="$1"

  [[ -e "${partial}" ]] || return 0

  if partial_is_held "${partial}"; then
    echo "Error: partial archive is still held by a running process:" >&2
    echo "  ${partial}" >&2
    exit 1
  fi

  if [[ "${RESET_PARTIAL}" != "1" ]]; then
    echo "Error: stale partial archive exists:" >&2
    echo "  ${partial}" >&2
    echo "Re-run with RESET_PARTIAL=1 after confirming the old tar stopped." >&2
    exit 1
  fi

  echo "Remove stale partial archive: ${partial}"
  rm -f -- "${partial}"
}

write_checksum() {
  local archive="$1"
  local checksum="${archive}.sha256"

  echo "Calculate SHA256: ${archive}"
  (
    cd "$(dirname "${archive}")"
    sha256sum "$(basename "${archive}")" > "$(basename "${checksum}")"
  )
}

package_method() {
  local label="$1"
  local source_base="$2"
  local source_train_member="$3"
  local source_val_member="$4"
  local archive_train_member="$5"
  local archive_val_member="$6"

  local train_path="${source_base}/${source_train_member}"
  local val_path="${source_base}/${source_val_member}"
  local archive="${ARCHIVE_ROOT}/${label}_train_val.tar"
  local partial="${archive}.partial"
  local train_count
  local val_count
  local train_bytes
  local val_bytes
  local total_bytes

  echo
  echo "============================================================"
  echo "Package method: ${label}"
  echo "============================================================"
  echo "Train source:   ${train_path}"
  echo "Val source:     ${val_path}"
  echo "Archive:        ${archive}"
  echo "Train member:   ${archive_train_member}"
  echo "Val member:     ${archive_val_member}"

  if [[ ! -d "${train_path}" ]]; then
    echo "Error: train directory not found: ${train_path}" >&2
    exit 1
  fi
  if [[ ! -d "${val_path}" ]]; then
    echo "Error: val directory not found: ${val_path}" >&2
    exit 1
  fi

  train_count="$(count_images "${train_path}")"
  val_count="$(count_images "${val_path}")"
  echo "Train images:   ${train_count}/${TRAIN_EXPECTED}"
  echo "Val images:     ${val_count}/${VAL_EXPECTED}"

  if [[ "${train_count}" -ne "${TRAIN_EXPECTED}" ]]; then
    echo "Error: ${label} train split is incomplete." >&2
    exit 1
  fi
  if [[ "${val_count}" -ne "${VAL_EXPECTED}" ]]; then
    echo "Error: ${label} val split is incomplete." >&2
    exit 1
  fi

  if [[ -f "${archive}" ]]; then
    if [[ "${REPLACE_ARCHIVE}" == 1 ]]; then
      if partial_is_held "${archive}"; then
        echo "Error: existing archive is held by a running process: ${archive}" >&2
        exit 1
      fi
      backup_root="${ARCHIVE_ROOT}/replaced_archives/$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${backup_root}"
      echo "Move replaced archive to: ${backup_root}"
      mv -- "${archive}" "${backup_root}/"
      [[ ! -e "${archive}.sha256" ]] || mv -- "${archive}.sha256" "${backup_root}/"
    else
      echo "Reuse existing final archive: ${archive}"
      if [[ ! -f "${archive}.sha256" ]]; then
        write_checksum "${archive}"
      fi
      return
    fi
  fi

  prepare_partial_path "${partial}"

  train_bytes="$(du -sb "${train_path}" | awk '{print $1}')"
  val_bytes="$(du -sb "${val_path}" | awk '{print $1}')"
  total_bytes=$((train_bytes + val_bytes))

  echo "Source size:    $(human_bytes "${total_bytes}")"
  echo "Started:        $(date)"
  echo "Temporary file: ${partial}"
  echo
  echo "tar is quiet while running. Monitor the .partial file from another shell."

  tar \
    -C "${source_base}" \
    --transform="s|^${source_train_member}|${archive_train_member}|" \
    --transform="s|^${source_val_member}|${archive_val_member}|" \
    -cf "${partial}" \
    "${source_train_member}" \
    "${source_val_member}"

  mv -- "${partial}" "${archive}"

  echo "Finished:       $(date)"
  ls -lh "${archive}"

  if [[ "${VERIFY_ARCHIVE}" == "1" ]]; then
    echo "Verify tar structure: ${archive}"
    tar -tf "${archive}" >/dev/null
  fi

  write_checksum "${archive}"
}

cat <<EOF
============================================================
Completed synthesis train+val packaging
============================================================
ARCHIVE_ROOT:    ${ARCHIVE_ROOT}
SYN_ROOT:        ${SYN_ROOT}
UWDF_ROOT:       ${UWDF_ROOT}
METHODS:         ${METHODS}
RESET_PARTIAL:   ${RESET_PARTIAL}
REPLACE_ARCHIVE: ${REPLACE_ARCHIVE}
VERIFY_ARCHIVE:  ${VERIFY_ARCHIVE}
Expected:        train=${TRAIN_EXPECTED}, val=${VAL_EXPECTED}
============================================================
EOF

for method in ${METHODS}; do
  case "${method}" in
    uwnr)
      package_method \
        uwnr \
        "${SYN_ROOT}" \
        uwnr_ruod_ref/generated/train \
        uwnr_ruod_ref/generated/val \
        uwnr_ruod_ref/generated/train \
        uwnr_ruod_ref/generated/val
      ;;
    syreanet)
      package_method \
        syreanet \
        "${SYN_ROOT}" \
        syreanet_synthesis/generated/train \
        syreanet_synthesis/generated/val \
        syreanet_synthesis/generated/train \
        syreanet_synthesis/generated/val
      ;;
    uwdf)
      package_method \
        uwdf \
        "${UWDF_ROOT}" \
        train \
        val \
        uwdf/generated/train \
        uwdf/generated/val
      ;;
    cut)
      package_method \
        cut \
        "${SYN_ROOT}" \
        cut/generated/train \
        cut/generated/val \
        cut/generated/train \
        cut/generated/val
      ;;
    watergan)
      package_method \
        watergan \
        "${SYN_ROOT}" \
        watergan/generated_step1564_official_mat/train \
        watergan/generated_step1564_official_mat/val \
        watergan/generated/train \
        watergan/generated/val
      ;;
    *)
      echo "Error: unsupported method in METHODS: ${method}" >&2
      exit 1
      ;;
  esac
done

echo
echo "Build combined checksum manifest."
: > "${ARCHIVE_ROOT}/SHA256SUMS.txt"
for method in ${METHODS}; do
  checksum="${ARCHIVE_ROOT}/${method}_train_val.tar.sha256"
  if [[ ! -f "${checksum}" ]]; then
    echo "Error: checksum missing: ${checksum}" >&2
    exit 1
  fi
done
find "${ARCHIVE_ROOT}" -maxdepth 1 -type f \
  -name '*_train_val.tar.sha256' -print0 \
  | sort -z \
  | xargs -0 -r cat \
  >> "${ARCHIVE_ROOT}/SHA256SUMS.txt"

echo
echo "============================================================"
echo "Packaging complete"
echo "============================================================"
ls -lh "${ARCHIVE_ROOT}"/*.tar "${ARCHIVE_ROOT}/SHA256SUMS.txt"
echo
du -ch "${ARCHIVE_ROOT}"/*.tar | tail -n 1
echo
cat "${ARCHIVE_ROOT}/SHA256SUMS.txt"
