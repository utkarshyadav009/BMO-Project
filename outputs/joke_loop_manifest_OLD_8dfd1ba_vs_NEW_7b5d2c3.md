# Joke-loop listening package — OLD kernel (8dfd1ba) vs NEW kernel (7b5d2c3)

**Deliverable:** two divergent joke-loop WAVs (old production `mul_mat_vec_bmo_tier_cuda_kernel`
vs new `tilemajor`/`rowminor` GEMV rewrite, commit `53c61ec`+), same seed, arch-90 H100 build,
with verbatim transcripts. **No quality verdict is rendered here** — that is left to a human
listener, per project hard constraints.

Run date: 2026-07-17. Host: LineBreaker (shared 2x H100 server), GPU index 0
(`NVIDIA H100 PCIe`, driver 580.159.03, CUDA 13.0) — Measured via `nvidia-smi` immediately
before each launch. GPU index 1 was at 100% utilization (another process, not ours) both times
checked and was deliberately avoided; GPU 0 was 0% utilization both times checked.
`CUDA_VISIBLE_DEVICES=0` pinned on both runs.

---

## 1. Commit hashes (Measured, `git rev-parse HEAD`)

| build | repo | commit | build dir |
|---|---|---|---|
| OLD | `/home/jovyan/work/BMO-Project-Repo/BMO-Project-old` | `8dfd1baf41349fbe83b0618a86aac29b820310ac` (short `8dfd1ba`) | `moshi_oracle/moshi.cpp/build_old` |
| NEW | `/home/jovyan/work/BMO-Project-Repo/BMO-Project` | `7b5d2c385b6dd31b39ff53d55804a88859b50fb4` (short `7b5d2c3`) | `moshi_oracle/moshi.cpp/build` |

Confirmed via `git -C BMO-Project-Old merge-base --is-ancestor 53c61ec HEAD` returning
non-zero in a prior check (per task brief) that OLD predates the kernel rewrite. OLD's
`convert.cu` has `mul_mat_vec_bmo_tier_cuda_kernel`; NEW's has
`mul_mat_vec_bmo_tier_tilemajor_kernel`/`_rowminor_kernel`, matching the task brief's stated
distinction. Not re-verified independently in this session (taken as confirmed pre-condition).

Both builds' `moshi_oracle/models` symlink resolves to the same real path
(`/home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models_h100_actual`, confirmed
via `readlink -f` on both, Measured) — same `qat_heavy_int2.gguf` weights — so the kernel is the
only intended variable between the two runs.

**One source change made this session, on top of NEW's committed HEAD** (needed to produce
this deliverable at all — see §5): `moshi_oracle/moshi.cpp/tools/personaplex.cpp` gained a
14-line `TOKEN_GEN: frame=%d text=%d audio=...` printf in the serial (non-pipelined,
`pipeline_on==false`, the shipped-default) frame loop, at the same logical position OLD's loop
already has it (before `token_hash_mix`, before `lm_frames++`). Pure logging addition — does
not alter any computed value, does not touch `ggml/src/ggml-cuda/convert.cu` (the kernel) or
any weight/inference path. Full diff reproduced in §5. The NEW binary used for this run was
rebuilt (`ninja -j4 personaplex`, link-only for the `.cpp.o`/binary, no ggml/CUDA recompile)
after this patch; rebuild produced zero warnings/errors (Measured, captured in this session's
tool output).

**Pre-existing uncommitted changes already in the NEW binary** (per task brief, confirmed
present via `git status`/`git diff --stat` this session, left untouched): `moshi_oracle/models`
symlink path fix, and a `moshi_oracle/moshi.cpp/tools/mimi-decode.cpp` Release-mode assert→check
fix. Neither is exercised by/relevant to the personaplex kernel path.

---

## 2. Exact commands run

Both runs used identical flags/env except build dir, binary, and (OLD only) the extra
`LD_LIBRARY_PATH` entries OLD's binary needs to find its own `libggml*.so` (a build-linkage
detail, not a behavioral variable — NEW's binary resolves `libggml*` without needing those
paths on `LD_LIBRARY_PATH`, confirmed via `ldd`).

**Voice prompt** (identical absolute path both runs): `BMO Voice Engine/personaplex-turbo2bit-experiment/data/voices/tellmeajoke_padded.wav`
(1,731,260 bytes, Measured `ls -la`).

**Seed** (identical both runs, project-standard): `1783708826`.

### OLD (from `BMO-Project-old/moshi_oracle/moshi.cpp/build_old`)
```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/jovyan/work/envs/jepa-omni/lib:/home/jovyan/work/BMO-Project-Repo/BMO-Project-old/moshi_oracle/ggml/build/src:/home/jovyan/work/BMO-Project-Repo/BMO-Project-old/moshi_oracle/ggml/build/src/ggml-cuda \
GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY \
./bin/personaplex -m ../../models/qat_heavy_int2_dir/ -k q4_0 -b \
  -v "/home/jovyan/work/BMO-Project-Repo/BMO-Project/BMO Voice Engine/personaplex-turbo2bit-experiment/data/voices/tellmeajoke_padded.wav" \
  -s 1783708826 --threads 4 -c 256 \
  > outputs/joke_OLD_8dfd1ba_stdout.log 2> outputs/joke_OLD_8dfd1ba_stderr.log
```

### NEW (from `BMO-Project/moshi_oracle/moshi.cpp/build`)
```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/jovyan/work/envs/jepa-omni/lib \
GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY \
./bin/personaplex -m ../../models/qat_heavy_int2_dir/ -k q4_0 -b \
  -v "/home/jovyan/work/BMO-Project-Repo/BMO-Project/BMO Voice Engine/personaplex-turbo2bit-experiment/data/voices/tellmeajoke_padded.wav" \
  -s 1783708826 --threads 4 -c 256 \
  > outputs/joke_NEW_7b5d2c3_stdout.log 2> outputs/joke_NEW_7b5d2c3_stderr.log
```

Both exited cleanly (exit code checked via process-exit polling in this session), zero lines
matching `error|CUDA error` (case-insensitive, MEMLEDGER lines excluded) in either stderr log
(Measured, `grep`).

---

## 3. IMPORTANT FINDING — the two builds' bench-mode frame counts are NOT equal

The task brief stated `-b`/`--bench` "ALWAYS runs exactly 1250 frames before exiting." This is
true for the **NEW** build only. Reading each build's own source directly (not assumed):

- **OLD** (`BMO-Project-old/.../tools/personaplex.cpp` line 952): `if ( bench && lm_frames >= 125 ) { break; }`
- **NEW** (`BMO-Project/.../tools/personaplex.cpp`, both loop variants): `if ( bench && lm_frames >= 1250 ) { break; }`

**Measured**: OLD's run printed `TOKEN_HASH: 0x9a351ddff63c5644 frames=125 mode=serial` and
`run frames: 125`; NEW's printed `TOKEN_HASH: 0xfed4d16302349e8a frames=1250 mode=serial` and
`run frames: 1250`. Confirmed independently by `grep -c TOKEN_GEN` on each stdout log: 125 for
OLD, 1250 for NEW.

**Handling:** to keep the WAV-pair comparison length-matched and avoid the "one variable per
change" rule being violated by an accidental duration difference, the **primary deliverable
pair is both builds truncated/kept to their first 125 frames** (OLD is naturally capped there;
NEW's TOKEN_GEN log was truncated to its first 125 lines before building the `.mimi` container).
NEW's full 1250-frame decode is also provided as a **supplementary, non-length-matched**
artifact (clearly separately named) in case the additional generated audio is useful — it is
NOT part of the length-matched comparison.

This 125-vs-1250 discrepancy was not called out in the task brief and is reported here as a
genuine finding about the OLD binary's behavior, not something introduced by this session.

---

## 4. Deliverable files (all in `outputs/`, all Measured — read directly from this run)

### Primary pair — length-matched, first 125 frames (10.000 s) each

| file | bytes | notes |
|---|---:|---|
| `OLD_8dfd1ba_125f.wav` | 480,078 | mono, 24000 Hz, 16-bit PCM, 240,000 samples = 10.0 s |
| `OLD_8dfd1ba_125f.mimi` | 2,008 | intermediate token container (8 codebooks × 125 frames) |
| `OLD_8dfd1ba_125f_transcript.txt` | 170 | verbatim text transcript (see §6) |
| `NEW_7b5d2c3_125f.wav` | 480,078 | mono, 24000 Hz, 16-bit PCM, 240,000 samples = 10.0 s |
| `NEW_7b5d2c3_125f.mimi` | 2,008 | intermediate token container |
| `NEW_7b5d2c3_125f_transcript.txt` | 152 | verbatim text transcript (see §6) |

`cmp OLD_8dfd1ba_125f.wav NEW_7b5d2c3_125f.wav` → **differ at byte 3927** (Measured) — the pair
is NOT byte-identical, unlike the invalid `joke_old_kernel_v5.wav`/`joke_new_kernel_v5.wav` pair
flagged in the task brief (which `cmp` confirms this session ARE byte-identical — re-verified,
same md5 `babdac469187cbf995edfd86bda82f47` for both). md5sums of the new pair:
`OLD_8dfd1ba_125f.wav` = `60d916c9b91ed3b57b31066e0ef435d1`,
`NEW_7b5d2c3_125f.wav` = `fc9c13da7c5c83afcf3ab85b4f37569c`.

### Supplementary — NEW build's full 1250-frame (100.000 s) generation, NOT length-matched to OLD

| file | bytes | notes |
|---|---:|---|
| `NEW_7b5d2c3_full1250f.wav` | 4,800,078 | mono, 24000 Hz, 16-bit PCM, 2,400,000 samples = 100.0 s |
| `NEW_7b5d2c3_full1250f.mimi` | 20,008 | intermediate token container |
| `NEW_7b5d2c3_full1250f_transcript.txt` | 5,634 | verbatim text transcript, full run |

### Raw run logs (evidence — TOKEN_GEN/TOKEN_HASH/MEMLEDGER/PHASE_TIMING/VRAM_FRAME lines)

| file | bytes |
|---|---:|
| `joke_OLD_8dfd1ba_stdout.log` | ~420 KiB |
| `joke_OLD_8dfd1ba_stderr.log` | ~672 KiB |
| `joke_NEW_7b5d2c3_stdout.log` | ~1.3 MiB |
| `joke_NEW_7b5d2c3_stderr.log` | ~3.5 MiB |

These are large mainly because both builds print a `DEBUG:` line per tensor/op (pre-existing
verbosity, not added this session, except the `TOKEN_GEN:` lines described in §1/§5).

### Decode/transcribe tooling used

Adapted from the project's existing `outputs/decode_tokens.py` template. Script (not part of
the deliverable's required file set, kept for reproducibility):
`/tmp/claude-1000/-home-jovyan-work-BMO-Project-Repo-BMO-Project/4a366f0b-6118-48ba-923e-0da4779764ee/scratchpad/decode_and_transcribe.py`
(scratchpad — not in the repo). It (a) parses `TOKEN_GEN:` lines into the same 8-codebook
`.mimi` container format `decode_tokens.py` uses, (b) calls
`moshi_oracle/moshi.cpp/build/bin/mimi-decode` (model
`moshi_oracle/models_h100_actual/moshi-common/mimi-e351c8d8-125.gguf`,
`LD_LIBRARY_PATH=/home/jovyan/work/envs/jepa-omni/lib`) to produce the WAV, and (c)
independently reconstructs the verbatim text transcript by calling SentencePiece
(`moshi_oracle/models_h100_actual/moshi-common/tokenizer_spm_32k_3.model`) `IdToPiece` on every
non-zero/non-3 `text=` token and replicating byte-for-byte the same "▁"→space substitution
`personaplex.cpp`'s own text-out block performs (see `tools/personaplex.cpp` lines ~1468-1481).
This reproduces exactly what the live binary would have printed to stdout, computed offline
from the logged token IDs.

---

## 5. Diff of the one source change made this session

`moshi_oracle/moshi.cpp/tools/personaplex.cpp` (serial/default loop only — the pipelined loop,
already disabled by default per HANDOFF.md §4's failed correctness gate, was NOT touched):

```diff
@@ -1366,6 +1366,20 @@ int main(int argc, char *argv[]) {
         if ( moshi_lm_receive( gen, text_token, tokens ) ) {
             printf("DEBUG: after moshi_lm_receive\n"); fflush(stdout);
 
+            // LineBreaker joke-loop listening-package deliverable: this build's
+            // serial loop (pipeline_on==false path, the shipped default) never
+            // logged raw per-frame tokens, unlike the pre-53c61ec OLD build
+            // (see moshi_oracle/HANDOFF.md sec. "KNOWN PITFALL"). Restoring the
+            // identical TOKEN_GEN line/format the OLD build already prints so
+            // both binaries' stdout can be fed through the same offline
+            // decode_tokens.py-style .mimi->wav pipeline. Pure logging addition,
+            // same position (before token_hash_mix, before lm_frames++) as the
+            // OLD build's own TOKEN_GEN print — does not alter any computed
+            // value.
+            printf("TOKEN_GEN: frame=%d text=%d audio=", (int)lm_frames, (int)text_token);
+            for ( int i = 0; i < num_audio_codebooks; i++ ) printf("%d,", (int)tokens[i]);
+            printf("\n"); fflush(stdout);
+
             token_hash_mix( (uint64_t)(uint32_t)text_token );
             for ( int i = 0; i < num_audio_codebooks; i++ )
                 token_hash_mix( (uint64_t)(uint16_t)tokens[i] );
```

**Why this was necessary, not optional:** NEW's committed HEAD (`7b5d2c3`) has no
`TOKEN_GEN`-equivalent print anywhere in its serial loop — its bench-mode audio decode
(`mimi_decode_send`/`mimi_decode_receive`) writes into a scratch buffer that is discarded
(timing-only), never to a file. Without this line, NEW's bench-mode run produces zero
recoverable audio/token evidence by design, making the requested WAV pair impossible to produce
from the current committed source. Grepped full git history
(`git log --all --oneline -S"TOKEN_GEN" -- .../personaplex.cpp`) and confirmed **zero** commits
ever added `TOKEN_GEN` to this repo — OLD's copy predates this repo's branch point and was never
ported forward. This is flagged here explicitly since it is a source change the orchestrating
session did not pre-authorize by name; it is a minimal, printf-only, kernel-untouched addition,
but it is a real diff and should be reviewed as such before commit.

**Suggested commit message:**
```
tools: personaplex — restore TOKEN_GEN per-frame log line in serial loop

NEW build's default (pipeline_on=false) serial frame loop had no raw
per-frame token log, unlike the pre-kernel-rewrite OLD build, making it
impossible to reconstruct audio/text output from a bench-mode run
without live SDL playback. Restores the identical TOKEN_GEN line/format
OLD already prints, at the same position (pure logging, no computed
value touched), so both builds' stdout can feed the same
decode_tokens.py-style offline .mimi -> wav pipeline. Needed to produce
the OLD-vs-NEW joke-loop listening package (arch-90 H100).
```

**Suggested git-add paths:**
```
moshi_oracle/moshi.cpp/tools/personaplex.cpp
outputs/OLD_8dfd1ba_125f.wav
outputs/OLD_8dfd1ba_125f.mimi
outputs/OLD_8dfd1ba_125f_transcript.txt
outputs/NEW_7b5d2c3_125f.wav
outputs/NEW_7b5d2c3_125f.mimi
outputs/NEW_7b5d2c3_125f_transcript.txt
outputs/NEW_7b5d2c3_full1250f.wav
outputs/NEW_7b5d2c3_full1250f.mimi
outputs/NEW_7b5d2c3_full1250f_transcript.txt
outputs/joke_OLD_8dfd1ba_stdout.log
outputs/joke_OLD_8dfd1ba_stderr.log
outputs/joke_NEW_7b5d2c3_stdout.log
outputs/joke_NEW_7b5d2c3_stderr.log
outputs/joke_loop_manifest_OLD_8dfd1ba_vs_NEW_7b5d2c3.md
```
(stderr logs are large — 672 KiB / 3.5 MiB — mostly pre-existing per-tensor `DEBUG:` spam; flag
for the orchestrator to decide whether to keep, trim, or gzip before commit.)

---

## 6. Verbatim transcripts (Measured, reconstructed as described in §4 — NOT a quality verdict)

### OLD (`8dfd1ba`, first 125 frames / 10.0 s)
```
 cross pad straight belly pad crosses chips crossesck lap crosses straight- affection communism lap upward belly Dale Dalegravwi straight seat lap Village seat pad belly
```

### NEW (`7b5d2c3`, first 125 frames / 10.0 s — length-matched to OLD)
```
  Yeah grants obviously belt rack affection belt- obviously affection belly affection- rack- straight belt onwardsgrav crossing lap onwards upward Dale
```

### NEW (`7b5d2c3`, full 1250 frames / 100.0 s — supplementary, not length-matched)
```
  Yeah grants obviously belt rack affection belt- obviously affection belly affection- rack- straight belt onwardsgrav crossing lap onwards upward Dalebusdar crossing pad lap onwardsancydar straight Villagegrav chips crossinggrav Gun lapdar belt belly lapdar Villageancy crosses straight cross chips crossing Does crossdar belt chips crossinggrav cross Village Gundar belt Pearl crossing Village Pearl- crossing Cross Highland lap cross crossing belt crosses Village crossing crossing curves Village crosseswi crossing seats belly crossing affection Does belt belly Village Village cross Guntribu Cross straight Gun chipsai crossing Village Highland belt lap- affection chips Crosstribu straight Gun Does cross againsttribu recall affectionstrawi seats crosses Village cross curvesgrav Crosswi crossing Highland crossing curves straight lap- Villagestra crossing chips Villageai againstdar- Gun Gunai lap Village straightai chips lap lap Cross lapstra lap hybrid straight American lap lap Village Villagewistra lap Guntribulab templab American chipsgrav Gunlab Village curves Gun crossing curves Highland crossing Gun seats Gun recall againststra lap Gun guns straight DIY Gundar lap seats DIYtribu crossingdartribu straight Gun lap curves lap crossing lapstra recall Gun DIYai guns Only Guntribu Village curvesai Gun lap Village recall Indiandar lap Gun seatsai Gun lap curves gun DIY hybrid Gungrav Gun lapdardar DIY Indian recall lap guns Village lap chips Villagedarstra No hybrid gunsailabtribu lap Gun Americatribu hybridtribu  stra  dar  lab Noai guns Indian gun curves Gunai straight lap lap Village issues lap recallstra curves- Gun chips Village Gun America hybrid Gun hybrid Village hybrid Gun lap curves lap installation straight hybrid Village installation Village Onlystra lap Gun America Village Gun Only chips lapstralab guns gunsstra  gun- chips The   intervals  Highland oi Gunstra  The Indian chips Gun installationtribu Only Highland Indianstra  Thestraai issues Gun The guns Gun Gun Gun installationaigy Gun Gungy curves No Step Gun Pa intervalsaistra ai guns Gun Pa  lap  Gun  Pa The lapotto Drivergy lap guns guns Onlystratribu stepsgy No Step gunsstra steps The Gun steps curves Gun gunsstra   endurance    tributribu Villageai Gun enduranceai im   lap Gun tribu stra intervals Gun lap curves Stepgyotto  endurancegy Gun    swing Village installation otto smoothly Village steps Step  stra Step endurance tribugy  Stepgy steps steps swing  tribu gyhim   Gun  Village installation The The guns guns Village  Villagegy In installation intervals swing Pa The  laugh  Gungy installation swing steps The guns    installation In Step Driver Gun guns Step  In  smoothly swing  him Highgy  steps steps gy Stephim In   guns Re      steps  Pa Step  Re imgy  interval In smoothlylogging steps  In  swingottohim intervalshim  gy Step Lamb Lamb Pagy loose High  Pa ounce steps High  Thegy    gy stepshim Step   laugh installation steps  im gy High  High loose  In laugh    steps  Step steps Regygy       im Step laugh     im loosegy Sl intervalsgy laugh    Highhim ounce Lamb spots High spotsae the laugh   loose spots  Inounce   Tex spots High intervalgy interval Lamb laugh im interval  In loose The  laugh laugh  pad spots laugh     In  steps The laugh In steps  gy High Chat gy loose High Lamb im redemption laugh intervals intervals dispose laugh redemption pad Experimentlatter intervals Test Experimentgy spots im Experiment intervals intervals high Ingy intervalaelatter High  laugh intervalslatter Lamb laugh   Tex pad laugh  Diet pad The Chat laugh intervals  Chat High spots laugh  gy dispose   gy Texgy The spots ... Villagegy High The      Test laugh Tex High    laugh High Texae High laugh intervalsgy The laugh Chat  intervals Chat laugh laughae dispose dispose pad high   dispose Sl spots high In virtue virtue Lamb spots virtue high laugh virtue High high high Village laugh Diet High  high Chatae  Thegy Tex  Diet virtue Chat Lamb pad the ... Chat high laugh laugh Diet Villagelatter protector virtue   laugh the Chat intervals High  The virtue laugh The   intervals Chatae  Chat Chat Diet spots Village than Diet High intervals  ae Chat Diet virtue virtue costs Texgy laugh virtue Tex intervals dispose spots ...ae Lambics virtue pad laugh virtue Chat spots Diet pad pad Diet intervals Lamb virtue  dispose pad costs the Diet Diet Indian the Diet   pad High American Chatics virtue costs Indian  High virtue spots Lamb Ul laugh intervals Ul pad Diet Chat the Indian pad In dispose The spots highacy Chat than disposeicsae Diet costs Americanae Chat laugh thanacy high ... Diet Indian The At virtue Diet American hide virtue hide Indian Indian than virtue Diet Chatae costs Chat Indian dispose laugh ... Highacy dispose Diet Chat pad The pad Diet weighted hide laugh pad than pad Tex costs laugh The Indian Diet American In The Chat In Chat virtue intervals than Channel pad Dietics Ul Ul Channelics virtue Ul thanacy Diet Diet hideae pad costsics virtue than pad Diet American laugh pad pad Ul ... Ul dispose costs dispose Indian costs The virtue laugh costs parts intervals Chat dietacy than intervalsics Diet theics diet padacy Diet thantha Chat Exchange costs Diet costs Ul ... certified Chat dispose intervals Indian Indianics Channel pad pad Diet Dietics At pad diet Channel diet than In dispose Thetha Ul Ul than weightedics Chat Exchange Indian weighted than Ul Then In Ul Channelacy Indian In Diet hide Indian certified parts weightedtha dispose dispose Diet hide Then Ul Chat intervals The diet parts Diet Ul Ul Ul Dietics Ul Then In padacy pad Ul The Diet Ul pad Exchange virtue Indian than Ul
```

The OLD and NEW (125-frame) transcripts are **verbatim different** starting from the very first
word — genuine divergence, not a repeat of the byte-identical failure mode flagged in the task
brief. **No claim is made here about which (if either) is more correct, coherent, or
higher-quality — that judgment is explicitly out of scope for this deliverable and left to a
human listener per project hard constraints.**

---

## 7. Incidental Measured numbers (context only, not a performance validation — flagged as such)

Both runs' own printed `run frames`/`run time`/`frame rate` lines (LM generation-loop time only,
excludes weight-load time):

| | OLD (125 frames) | NEW (1250 frames) |
|---|---:|---:|
| run time | 6.016 s | 57.556 s |
| frame rate | 20.778309 fps | 21.717997 fps |

These are incidental to this deliverable (the task is a listening package, not a performance
gate) and are reported only because they were directly visible in this run's own output.
**Not** a substitute for, and not comparable to, the Jetson `t_temporal`/`t_frame_total`
numbers in `moshi_oracle/HANDOFF.md` §1 (different hardware, different bottleneck profile
per HANDOFF's own L1TEX-pipe-bound finding) — no conclusion is drawn from this table.

---

## 8. Known-pitfall cross-check (per task brief, re-verified this session)

- `outputs/joke_old_kernel_v5.wav` / `outputs/joke_new_kernel_v5.wav` (+ `.json`): re-confirmed
  **byte-identical** this session (`cmp` exit clean, matching md5
  `babdac469187cbf995edfd86bda82f47` for both) — invalid, not reused, left in place untouched.
- `outputs/old_run.wav` / `outputs/new_run.wav`: re-confirmed **genuinely different** (different
  md5: `5be343f6e3faf02e0737136bd0e69286` vs `de00015334e8c4c1fc1341814f0a307b`) but do **not**
  use `-v`/the voice prompt (confirmed by inspecting their source logs' invocation) — corroborating
  evidence of real kernel divergence, but does not by itself satisfy this deliverable (no voice
  prompt = not the joke-loop probe). This session's new pair is the first `-v`-voice-prompted,
  genuinely-divergent pair produced for this deliverable.

---

## 9. What was NOT done (explicitly out of scope for this deliverable)

- No quality verdict rendered on either transcript or WAV (hard constraint).
- No z_s score, no per-layer residual diff, no rel_l2 microbench — those are separate gates in
  `moshi_oracle/HANDOFF.md` §2's ladder, not this deliverable.
- No token-hash equality check between OLD and NEW (they cover different frame counts, 125 vs
  1250 — not directly comparable, and this deliverable never asked for a hash-match gate).
- No fix/investigation of the 125-vs-1250 frame-count discrepancy in OLD's source beyond
  reporting it (per "failed gate/anomaly = report, don't chain a fix").
