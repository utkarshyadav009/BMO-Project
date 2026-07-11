#!/usr/bin/env bash
# jetson_preflight.sh — gate Jetson measurement runs on real memory contiguity,
# not just totals. Must be run as root (uses /proc/sys/vm/drop_caches and
# /proc/sys/vm/compact_memory). A run whose preflight does not PASS is not a
# valid data point — do not report numbers from it.
#
# Finding this exists to encode: Jetson cudaMalloc/NvMap allocations fail on
# CONTIGUITY, not totals. A request well under the reported free total can
# still fail with NvMap error 12 if the largest contiguous free block (tegrastats
# "lfb") is too small. "free" and "available" are not sufficient signals by
# themselves on this platform.
#
# GATE NOTE: tegrastats "lfb NxSIZE" reports the buddy allocator's max-order
# free block count/size. On aarch64 with 4 KiB pages, MAX_ORDER caps the block
# size at 4 MiB — SIZE will essentially always read 4MB regardless of actual
# fragmentation state; it is architecturally incapable of reporting a bigger
# block, so gating on N*SIZE (as an earlier version of this script did) is
# gating on something that can never exceed ~4MiB*N anyway, and worse, made
# the gate look like it wanted a single >=512MiB contiguous block, which the
# allocator cannot report even when memory is genuinely healthy. Gate on N
# (count of max-order blocks) directly instead: N>=128 means >=512MiB is held
# in max-order blocks, which is the actual signal that matters.
#
# Usage: sudo bash tools/jetson_preflight.sh
# Exit 0 + "PREFLIGHT: PASS" on success, exit 1 + "PREFLIGHT: FAIL" otherwise.

set -euo pipefail

MIN_LFB_N=128
MIN_FREE_MIB=5500
LOG_FILE="${LOG_FILE:-$(dirname "$0")/jetson_preflight.log}"

if [ "$(id -u)" -ne 0 ]; then
    echo "PREFLIGHT: FAIL (must run as root — needs drop_caches/compact_memory)" >&2
    exit 1
fi

# Every run appends here (timestamped) — a run without a matching PASS entry
# in this log is not a valid data point. exec redirects the REST of this
# script to both stdout and the log; the root-check above (before this line)
# deliberately isn't logged, since a non-root invocation is a usage error,
# not a measurement.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Compaction BEFORE sampling — the whole point is to measure the
# post-compaction state, not to compact based on what we measured.
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
sleep 5

# NOTE: `tegrastats | head -1` under `pipefail` fails spuriously — head closing
# the pipe after one line sends tegrastats SIGPIPE, and pipefail propagates that
# as a nonzero pipeline status, which set -e then treats as a hard failure
# before any output is printed. Decouple via a temp file instead.
TEGRA_TMP=$(mktemp)
timeout 3 tegrastats --interval 1000 > "$TEGRA_TMP" 2>&1 || true
SAMPLE=$(head -1 "$TEGRA_TMP")
rm -f "$TEGRA_TMP"
echo "PREFLIGHT_TEGRASTATS_SAMPLE: ${SAMPLE}"

echo "PREFLIGHT_BUDDYINFO:"
cat /proc/buddyinfo

# Parse "RAM <used>/<total>MB (lfb <N>x<SIZE>MB)"
USED_MIB=$(echo "$SAMPLE" | grep -oP 'RAM \K[0-9]+(?=/[0-9]+MB)')
TOTAL_MIB=$(echo "$SAMPLE" | grep -oP 'RAM [0-9]+/\K[0-9]+(?=MB)')
LFB_N=$(echo "$SAMPLE" | grep -oP 'lfb \K[0-9]+(?=x)')
LFB_BLOCK_MIB=$(echo "$SAMPLE" | grep -oP 'lfb [0-9]+x\K[0-9]+(?=MB)')

if [ -z "$USED_MIB" ] || [ -z "$TOTAL_MIB" ] || [ -z "$LFB_N" ] || [ -z "$LFB_BLOCK_MIB" ]; then
    echo "PREFLIGHT: FAIL (could not parse tegrastats sample: ${SAMPLE})" >&2
    exit 1
fi

FREE_MIB=$(( TOTAL_MIB - USED_MIB ))

echo "PREFLIGHT_PARSED: free_MiB=${FREE_MIB} total_MiB=${TOTAL_MIB} lfb_N=${LFB_N} lfb_block_MiB=${LFB_BLOCK_MIB}"
echo "PREFLIGHT_GATE: require free_MiB>=${MIN_FREE_MIB} AND lfb_N>=${MIN_LFB_N}"

FAIL_REASONS=()
if [ "$FREE_MIB" -lt "$MIN_FREE_MIB" ]; then
    FAIL_REASONS+=("free_MiB=${FREE_MIB} < ${MIN_FREE_MIB}")
fi
if [ "$LFB_N" -lt "$MIN_LFB_N" ]; then
    FAIL_REASONS+=("lfb_N=${LFB_N} < ${MIN_LFB_N}")
fi

if [ ${#FAIL_REASONS[@]} -eq 0 ]; then
    echo "PREFLIGHT: PASS"
    exit 0
else
    IFS='; '
    echo "PREFLIGHT: FAIL (${FAIL_REASONS[*]})"
    exit 1
fi
