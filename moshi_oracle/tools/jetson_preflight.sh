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
# GATE NOTE (revised after direct measurement on this hardware): tegrastats'
# "lfb NxSIZE" field reports only order-10 (4 MiB) blocks, capped there
# regardless of what's actually free at higher orders — on THIS kernel,
# /proc/buddyinfo shows orders up to 12 (16 MiB) are tracked and populated,
# so tegrastats' lfb is not the ground truth here and gating on it produces
# false FAILs on genuinely healthy systems (confirmed directly: a fresh
# reboot showed tegrastats lfb_N=22 — apparently unhealthy — while
# /proc/buddyinfo's Normal zone alone had 237 order-12 blocks, ~3.7 GB, free).
# Gate on /proc/buddyinfo directly instead: sum bytes held in blocks of
# order>=10 (4 MiB+) across all zones — this is the real signal tegrastats
# was trying (and failing) to approximate.
#
# Usage: sudo bash tools/jetson_preflight.sh
# Exit 0 + "PREFLIGHT: PASS" on success, exit 1 + "PREFLIGHT: FAIL" otherwise.

set -euo pipefail

PAGE_SIZE_BYTES=$(getconf PAGESIZE)
MIN_LARGE_BLOCK_MIB=512
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

# Parse "RAM <used>/<total>MB" for the free-total check (tegrastats totals are
# reliable; it's specifically the lfb fragmentation field that's misleading).
USED_MIB=$(echo "$SAMPLE" | grep -oP 'RAM \K[0-9]+(?=/[0-9]+MB)')
TOTAL_MIB=$(echo "$SAMPLE" | grep -oP 'RAM [0-9]+/\K[0-9]+(?=MB)')

if [ -z "$USED_MIB" ] || [ -z "$TOTAL_MIB" ]; then
    echo "PREFLIGHT: FAIL (could not parse tegrastats sample: ${SAMPLE})" >&2
    exit 1
fi

FREE_MIB=$(( TOTAL_MIB - USED_MIB ))

# Sum bytes held in blocks of order>=10 (4 MiB+) across all zones in
# /proc/buddyinfo. Each "zone" line: "Node N, zone NAME <count_order0>
# <count_order1> ... <count_orderMAX>" — order K block = PAGE_SIZE * 2^K.
LARGE_BLOCK_MIB=$(awk -v page="$PAGE_SIZE_BYTES" '
    /zone/ {
        for (i = 5; i <= NF; i++) {
            order = i - 5
            if (order >= 10) {
                total += $i * page * (2 ^ order)
            }
        }
    }
    END { printf "%.0f", total / 1024 / 1024 }
' /proc/buddyinfo)

echo "PREFLIGHT_PARSED: free_MiB=${FREE_MIB} total_MiB=${TOTAL_MIB} large_block_MiB(order>=10, all zones)=${LARGE_BLOCK_MIB}"
echo "PREFLIGHT_GATE: require free_MiB>=${MIN_FREE_MIB} AND large_block_MiB>=${MIN_LARGE_BLOCK_MIB}"

FAIL_REASONS=()
if [ "$FREE_MIB" -lt "$MIN_FREE_MIB" ]; then
    FAIL_REASONS+=("free_MiB=${FREE_MIB} < ${MIN_FREE_MIB}")
fi
if [ "$LARGE_BLOCK_MIB" -lt "$MIN_LARGE_BLOCK_MIB" ]; then
    FAIL_REASONS+=("large_block_MiB=${LARGE_BLOCK_MIB} < ${MIN_LARGE_BLOCK_MIB}")
fi

if [ ${#FAIL_REASONS[@]} -eq 0 ]; then
    echo "PREFLIGHT: PASS"
    exit 0
else
    IFS='; '
    echo "PREFLIGHT: FAIL (${FAIL_REASONS[*]})"
    exit 1
fi
