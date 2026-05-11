# Online θ Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an on-device controller that monotonically raises `FuzzyNumber` during PA when the residual graph is in active-peeling phase, layered on top of the existing static v3 predictor.

**Architecture:** Single-thread block-0 controller appended inside the existing `if (grid.thread_rank() == 0) { ... }` block at the iteration boundary of `P_SL_ELS_SDC_CTA_S` (PA.cu lines 864–871). Compile-gated via `-DDYNAMIC_THETA` so the controller is zero-overhead when off. Trajectory `(iter, new_θ)` logged to a fixed-size device array, dumped to stdout + optional JSON post-PA. 5 new CHROMA CLI flags wire to a new `setDynamicParameters` host launcher.

**Tech Stack:** CUDA (cooperative_groups grid.sync), C++17, m2cgen-emitted RF model (unchanged), Python for sweep / aggregation.

**Branch:** `theta-predictor-online`. Predecessor: static v3 predictor on `main` at commit `a4ec1b6`.

---

## File Structure

| file | role | lines added (est.) |
|------|------|-------------------|
| `CHROMA/globals.cu` | new __device__ state (`last_remove_size`, `g_nodes`, `bump_*`, `CTRL_*`) gated by `#ifdef DYNAMIC_THETA` | ~25 |
| `CHROMA/globals.cuh` | extern declarations of the above | ~20 |
| `CHROMA/chroma_utils.cu` | new `setDynamicParameters_kernel` + host wrapper; extend `resetForRun()` to clear bump state | ~30 |
| `CHROMA/chroma_utils.cuh` | declaration of `setDynamicParameters` | ~5 |
| `CHROMA/PA.cu` | controller block inside the existing single-thread iteration boundary in `P_SL_ELS_SDC_CTA_S` | ~20 |
| `CHROMA/CHROMA.cu` | 5 new CLI flags, host-side dynamic param dispatch, post-PA trajectory readback + stdout print + optional JSON | ~80 |
| `CHROMA/Makefile` | `DYNAMIC_THETA ?= 0` build flag | ~4 |
| `scripts/sweep_dynamic_theta.py` | three-way EGR sweep: static_v3 / dyn_only / static + dyn | NEW (~150) |

All `globals.cu`, `globals.cuh`, `chroma_utils.cu`, `PA.cu` edits are wrapped in `#ifdef DYNAMIC_THETA` so a build with `DYNAMIC_THETA=0` (the default) is byte-identical to the current main.

---

## Task 1: Device state + globals (compile-gated)

**Files:**
- Modify: `CHROMA/globals.cu`
- Modify: `CHROMA/globals.cuh`

- [ ] **Step 1: Add device state to `CHROMA/globals.cu`**

Append (after the existing `__device__ int FuzzyNumber = 0;`):

```cpp
#ifdef DYNAMIC_THETA
// ─── Online θ controller state (compile-gated) ─────────────────────
// Number of vertices in graph (set once before kernel by setDynamicParameters)
__device__ int g_nodes = 0;

// Number of vertices already removed at the last controller checkpoint.
// Initialised to 0; updated every CTRL_K iterations.
__device__ int last_remove_size = 0;

// Tunables (set by setDynamicParameters before each kernel launch).
__device__ int   CTRL_K               = 0;       // 0 disables controller at runtime
__device__ float CTRL_RATE_THRESHOLD  = 0.0f;    // bump trigger (fraction of V removed per iter)
__device__ int   CTRL_STEP            = 1;       // amount to bump FuzzyNumber
__device__ int   CTRL_CAP             = 0;       // upper bound on FuzzyNumber (0 = no cap)

// Trajectory log (single-writer: block 0 thread 0, race-free).
#define BUMP_LOG_MAX 32
__device__ int bump_count = 0;
__device__ int bump_iter [BUMP_LOG_MAX] = {0};
__device__ int bump_theta[BUMP_LOG_MAX] = {0};
#endif
```

- [ ] **Step 2: Add extern declarations to `CHROMA/globals.cuh`**

Append (after the existing `extern __device__ int FuzzyNumber;`):

```cpp
#ifdef DYNAMIC_THETA
extern __device__ int   g_nodes;
extern __device__ int   last_remove_size;
extern __device__ int   CTRL_K;
extern __device__ float CTRL_RATE_THRESHOLD;
extern __device__ int   CTRL_STEP;
extern __device__ int   CTRL_CAP;

#define BUMP_LOG_MAX 32
extern __device__ int   bump_count;
extern __device__ int   bump_iter [BUMP_LOG_MAX];
extern __device__ int   bump_theta[BUMP_LOG_MAX];
#endif
```

- [ ] **Step 3: Sanity-compile `globals.cu` with the new flag**

Run from worktree root:
```bash
nvcc -arch=sm_86 -DDYNAMIC_THETA -dc CHROMA/globals.cu -o /tmp/globals_dyn.o
ls -la /tmp/globals_dyn.o
```
Expected: object file produced, no warnings about the new symbols.

Then verify the **off** path is still clean:
```bash
nvcc -arch=sm_86 -dc CHROMA/globals.cu -o /tmp/globals_off.o
nm /tmp/globals_off.o | grep -E "g_nodes|last_remove_size|bump_" || echo "(no dyn symbols — correct)"
```
Expected: no `g_nodes`/`last_remove_size`/`bump_*` symbols when `-DDYNAMIC_THETA` is absent.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/globals.cu CHROMA/globals.cuh
git commit -m "online-θ (T1): add compile-gated controller state in globals"
```

---

## Task 2: `setDynamicParameters` host launcher

**Files:**
- Modify: `CHROMA/chroma_utils.cu`
- Modify: `CHROMA/chroma_utils.cuh`

- [ ] **Step 1: Add kernel + host wrapper to `CHROMA/chroma_utils.cu`**

Append (anywhere in the file; recommended near the existing `setParameters` definition):

```cpp
#ifdef DYNAMIC_THETA
__global__ void setDynamicParameters_kernel(
    int   nodes_in,
    int   K,
    float rate,
    int   step,
    int   cap)
{
    g_nodes              = nodes_in;
    CTRL_K               = K;
    CTRL_RATE_THRESHOLD  = rate;
    CTRL_STEP            = step;
    CTRL_CAP             = cap;
    last_remove_size     = 0;
    bump_count           = 0;
}

void setDynamicParameters(int nodes, int K, float rate, int step, int cap)
{
    setDynamicParameters_kernel<<<1, 1>>>(nodes, K, rate, step, cap);
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif
```

- [ ] **Step 2: Declare in `CHROMA/chroma_utils.cuh`**

Append (after the existing `void setParameters(...)` declaration if any, or near the top of the function declarations):

```cpp
#ifdef DYNAMIC_THETA
void setDynamicParameters(int nodes, int K, float rate, int step, int cap);
#endif
```

- [ ] **Step 3: Sanity-build (CHROMA full target with flag)**

```bash
cd CHROMA && make clean > /dev/null 2>&1
make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -3
```
Expected: clean build (Task 4 will add the Makefile flag wiring; for this step, manually pass `-DDYNAMIC_THETA` via env if needed).

If Makefile doesn't yet recognise `DYNAMIC_THETA=1`, do quick override:
```bash
cd CHROMA && make clean > /dev/null 2>&1
DC_FLAGS_EXTRA=-DDYNAMIC_THETA CXXFLAGS_EXTRA=-DDYNAMIC_THETA make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -3
```
Expected: link succeeds; if the Makefile doesn't honour `*_EXTRA`, defer this verification to Task 4 (Makefile flag wiring), where we will rebuild correctly.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/chroma_utils.cu CHROMA/chroma_utils.cuh
git commit -m "online-θ (T2): setDynamicParameters host launcher"
```

---

## Task 3: Extend `resetForRun()` to clear bump state

**Files:**
- Modify: `CHROMA/chroma_utils.cu` (existing `resetForRun()`)

- [ ] **Step 1: Inspect current `resetForRun()` body**

```bash
grep -n -A20 "^void resetForRun" CHROMA/chroma_utils.cu | head -25
```
Note the function signature and current body to know where to insert.

- [ ] **Step 2: Add controller-state reset block**

Inside `resetForRun(const ECLgraph& g, DevPtr& d)`, append (just before the function's closing brace):

```cpp
#ifdef DYNAMIC_THETA
    // Reset dynamic-θ controller state for the new run.
    // (host-side memset-equivalent of the device variables we touch each run)
    int   zero_int = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(last_remove_size, &zero_int,  sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(bump_count,       &zero_int,  sizeof(int)));
#endif
```

- [ ] **Step 3: Sanity-build**

```bash
cd CHROMA && make clean > /dev/null 2>&1
make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -3
```
Expected: clean build with `DYNAMIC_THETA` undefined; `resetForRun()` is unchanged behaviourally.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/chroma_utils.cu
git commit -m "online-θ (T3): reset controller state in resetForRun()"
```

---

## Task 4: Makefile `DYNAMIC_THETA` build flag

**Files:**
- Modify: `CHROMA/Makefile`

- [ ] **Step 1: Add the flag definition + propagation**

After the existing `PRE_MODEL ?= 0` line (around line 11), add:

```makefile
DYNAMIC_THETA ?= 0     # 0: off (zero overhead); 1: enable on-device θ controller
```

In the existing `ifeq ($(PRE_MODEL), 1)` block (around line 35), AFTER the existing `DC_FLAGS += -DPRED_MODEL -I../model` lines, append:

```makefile
ifeq ($(DYNAMIC_THETA), 1)
    DC_FLAGS += -DDYNAMIC_THETA
    CXXFLAGS += -DDYNAMIC_THETA
endif
```

- [ ] **Step 2: Add `DYNAMIC_THETA` to the Usage comment**

Update the `# ================== Usage ==================` block at the top of the Makefile to include:

```makefile
# $ make PRE_MODEL=1 DYNAMIC_THETA=1   Enable predict model + on-device θ controller
```

- [ ] **Step 3: Verify both build paths**

```bash
cd CHROMA
make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -2
make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -2
```
Both should succeed cleanly.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/Makefile
git commit -m "online-θ (T4): Makefile DYNAMIC_THETA=1 build flag"
```

---

## Task 5: Smoke-test that DYNAMIC_THETA=0 build is byte-identical

**Files:** none (verification only)

- [ ] **Step 1: Build current main + a test invocation**

```bash
cd /home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -2
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "EGC|colors used|verif|Iter count" > /tmp/dyntheta_off.txt
cat /tmp/dyntheta_off.txt
```
Expected: `EGC θ: 3 (Predicted)`, `colors used: 73`, `result verification passed`, `Iter count: <some value>`.

- [ ] **Step 2: Build with DYNAMIC_THETA=1 + same invocation (no --dynamic-theta CLI)**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -2
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "EGC|colors used|verif|Iter count" > /tmp/dyntheta_on_inactive.txt
cat /tmp/dyntheta_on_inactive.txt
```
Expected: identical output to step 1 (controller is compiled in but `CTRL_K=0` means it never triggers, no behaviour change).

- [ ] **Step 3: Diff**

```bash
diff /tmp/dyntheta_off.txt /tmp/dyntheta_on_inactive.txt
```
Expected: no output (files identical).

- [ ] **Step 4: Restore canonical build for downstream tasks**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -1
cd ..
```

(No commit — this is a verification step only.)

---

## Task 6: Insert controller block into `P_SL_ELS_SDC_CTA_S`

**Files:**
- Modify: `CHROMA/PA.cu` (existing single-thread block at lines ~864–871 inside `P_SL_ELS_SDC_CTA_S`)

- [ ] **Step 1: Locate the exact insertion point**

```bash
grep -n "iteration = iteration + 1 + FuzzyNumber" CHROMA/PA.cu | head -10
```
You'll see multiple matches (one per PA variant). The one in `P_SL_ELS_SDC_CTA_S` is around line 869 (verify by checking the surrounding kernel header at line 536 with `awk 'NR>=865 && NR<=872 {print NR": "$0}' CHROMA/PA.cu`).

The existing block looks like:

```cpp
if (grid.thread_rank() == 0) {
    worker += remove_size;
    remove_size = 0;
    theta = g_minDegree;
    atomicExch(&g_minDegree, 0x7FFFFFFF);
    iteration = iteration + 1 + FuzzyNumber;
    iter_count++;
}
```

- [ ] **Step 2: Replace it with the controller-augmented version**

Use the Edit tool to replace exactly the 7-line block above with:

```cpp
if (grid.thread_rank() == 0) {
    worker += remove_size;
    remove_size = 0;
    theta = g_minDegree;
    atomicExch(&g_minDegree, 0x7FFFFFFF);
    iteration = iteration + 1 + FuzzyNumber;
    iter_count++;

#ifdef DYNAMIC_THETA
    // ─── Online θ controller ─────────────────────────────────────
    // Every CTRL_K iterations, check removal rate. If we removed at least
    // CTRL_RATE_THRESHOLD·V vertices since the last checkpoint, bump
    // FuzzyNumber by CTRL_STEP (capped at CTRL_CAP). Monotone non-decreasing.
    if (CTRL_K > 0 && (iter_count % CTRL_K) == 0 && iter_count >= CTRL_K) {
        int delta = worker - last_remove_size;
        last_remove_size = worker;
        if (delta >= (int)(CTRL_RATE_THRESHOLD * (float)g_nodes)) {
            int new_fz = FuzzyNumber + CTRL_STEP;
            if (CTRL_CAP > 0 && new_fz > CTRL_CAP) new_fz = CTRL_CAP;
            if (new_fz > FuzzyNumber) {
                FuzzyNumber = new_fz;
                if (bump_count < BUMP_LOG_MAX) {
                    bump_iter [bump_count] = iter_count;
                    bump_theta[bump_count] = new_fz;
                    bump_count = bump_count + 1;
                }
            }
        }
    }
#endif
}
```

The trailing `grid.sync();` (existing line 872) ensures all threads see the updated `FuzzyNumber` before the next iteration's body. No new sync needed.

- [ ] **Step 3: Build with DYNAMIC_THETA=1**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -3
```
Expected: clean build. The new `g_nodes`, `last_remove_size`, `CTRL_*`, `bump_*` symbols all resolve through globals.cu.

- [ ] **Step 4: Re-run smoke from Task 5 to confirm no regression**

```bash
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "EGC|colors used|verif|Iter count"
cd ..
```
Expected: `EGC θ: 3 (Predicted)`, `colors used: 73`, `result verification passed`. Identical to Task 5 because `CTRL_K=0` (no `setDynamicParameters` call yet).

- [ ] **Step 5: Commit**

```bash
git add CHROMA/PA.cu
git commit -m "online-θ (T6): controller block in P_SL_ELS_SDC_CTA_S iteration boundary"
```

---

## Task 7: CHROMA CLI flags

**Files:**
- Modify: `CHROMA/CHROMA.cu` (CLI parsing only — wiring to setDynamicParameters comes in Task 8)

- [ ] **Step 1: Locate `print_help()` and the argv parsing loop**

```bash
grep -n "print_help\|argv\[i\]" CHROMA/CHROMA.cu | head -20
```
`print_help()` is around line 32; argv loop is around line 180.

- [ ] **Step 2: Add help text**

In `print_help()`, after the existing `--predict` help line (~line 54), insert:

```cpp
    std::cout << "  --dynamic-theta           Enable on-device θ controller (requires DYNAMIC_THETA=1 build)\n";
    std::cout << "  --dynamic-K <int>         Sample interval, iterations between checks (default 10)\n";
    std::cout << "  --dynamic-rate <float>    Trigger threshold = fraction of V removed per iter (default 0.005)\n";
    std::cout << "  --dynamic-step <int>      Bump amount per trigger (default 1)\n";
    std::cout << "  --dynamic-cap <int>       Max FuzzyNumber (default θ_initial + 5)\n";
    std::cout << "  --dynamic-log <path>      Append trajectory JSON to <path> (default no log)\n";
```

- [ ] **Step 3: Add CLI variables in `main()`**

Just after the existing `bool use_predicted_elastic = false;` declaration in `main()`, add:

```cpp
    bool        dynamic_theta = false;
    int         dynamic_K     = 10;
    float       dynamic_rate  = 0.005f;
    int         dynamic_step  = 1;
    int         dynamic_cap   = 0;        // 0 = "θ_initial + 5" (resolved later)
    std::string dynamic_log;
```

- [ ] **Step 4: Parse the new flags**

Inside the argv loop, just after the existing `else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--predict") == 0) { use_predicted_elastic = true; }`, append:

```cpp
        } else if (strcmp(argv[i], "--dynamic-theta") == 0) {
            dynamic_theta = true;
        } else if (strcmp(argv[i], "--dynamic-K") == 0) {
            if (i + 1 >= argc) { std::cerr << "Error: --dynamic-K needs an int.\n"; return 1; }
            dynamic_K = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--dynamic-rate") == 0) {
            if (i + 1 >= argc) { std::cerr << "Error: --dynamic-rate needs a float.\n"; return 1; }
            dynamic_rate = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--dynamic-step") == 0) {
            if (i + 1 >= argc) { std::cerr << "Error: --dynamic-step needs an int.\n"; return 1; }
            dynamic_step = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--dynamic-cap") == 0) {
            if (i + 1 >= argc) { std::cerr << "Error: --dynamic-cap needs an int.\n"; return 1; }
            dynamic_cap = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--dynamic-log") == 0) {
            if (i + 1 >= argc) { std::cerr << "Error: --dynamic-log needs a path.\n"; return 1; }
            dynamic_log = argv[++i];
```

- [ ] **Step 5: Sanity-build**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -2
./CHROMA --help 2>&1 | grep -E "dynamic|Usage" | head -8
cd ..
```
Expected: build clean; help shows the 6 new dynamic options.

- [ ] **Step 6: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "online-θ (T7): CLI flags --dynamic-theta + tunables (parsing only)"
```

---

## Task 8: Wire CLI to `setDynamicParameters` + warn when compile flag missing

**Files:**
- Modify: `CHROMA/CHROMA.cu`

- [ ] **Step 1: Locate the existing `setParameters<<<>>>` call**

```bash
grep -n "setParameters" CHROMA/CHROMA.cu | head -5
```
The host-side `setParameters<<<1,1>>>(fuzzy_number);` call is around line 352.

- [ ] **Step 2: Add dispatch immediately after it**

Just after the line `setParameters<<<1, 1>>>(fuzzy_number);`, add:

```cpp
    // Dynamic θ controller setup (no-op when --dynamic-theta isn't set).
    if (dynamic_theta) {
#ifdef DYNAMIC_THETA
        int cap = (dynamic_cap > 0) ? dynamic_cap : (fuzzy_number + 5);
        setDynamicParameters(g.nodes, dynamic_K, dynamic_rate, dynamic_step, cap);
        printf("Dynamic θ: K=%d  rate=%.4f  step=%d  cap=%d  initial=%d\n",
               dynamic_K, dynamic_rate, dynamic_step, cap, fuzzy_number);
#else
        std::cerr << "Warning: --dynamic-theta has no effect — rebuild with "
                     "`make ... DYNAMIC_THETA=1`. Falling back to static θ.\n";
#endif
    }
```

(`g.nodes` is the loaded ECLgraph's node count — already in scope at this point in `main()`.)

- [ ] **Step 3: Smoke — verify warning prints in DYNAMIC_THETA=0 build**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -1
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict --dynamic-theta 2>&1 | grep -E "Warning|EGC|colors used"
cd ..
```
Expected: `Warning: --dynamic-theta has no effect — rebuild with...`, then `EGC θ: 3 (Predicted)`, `colors used: 73` (static behaviour).

- [ ] **Step 4: Smoke — verify dispatch fires in DYNAMIC_THETA=1 build**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 DYNAMIC_THETA=1 2>&1 | tail -1
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict --dynamic-theta 2>&1 | grep -E "Dynamic θ|EGC|colors used|verif"
cd ..
```
Expected: `Dynamic θ: K=10  rate=0.0050  step=1  cap=8  initial=3`, then `EGC θ: 3 (Predicted)`, `colors used: 73` (or close — controller may or may not bump on facebook depending on iter count).

- [ ] **Step 5: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "online-θ (T8): wire --dynamic-theta to setDynamicParameters + warn fallback"
```

---

## Task 9: Trajectory readback + stdout print

**Files:**
- Modify: `CHROMA/CHROMA.cu` (post-PA, after the kernel returns)

- [ ] **Step 1: Locate the post-PA reporting section**

```bash
grep -n "colors used:\|Iter count" CHROMA/CHROMA.cu | head -5
```
Find where the per-run stats are printed. Insertion point is right after that block (so trajectory appears with the run summary).

- [ ] **Step 2: Add readback + print block**

Just after the existing `printf("Iter count: ...")` style line (or wherever the per-run summary ends), add:

```cpp
#ifdef DYNAMIC_THETA
    if (dynamic_theta) {
        int        bump_n = 0;
        int        bumps_iter [BUMP_LOG_MAX]  = {0};
        int        bumps_theta[BUMP_LOG_MAX]  = {0};
        cudaMemcpyFromSymbol(&bump_n,      bump_count, sizeof(int));
        cudaMemcpyFromSymbol(bumps_iter,   bump_iter,  sizeof(bumps_iter));
        cudaMemcpyFromSymbol(bumps_theta,  bump_theta, sizeof(bumps_theta));

        printf("θ trajectory: start=%d  bumps=[", fuzzy_number);
        for (int b = 0; b < bump_n; ++b) {
            if (b > 0) printf(", ");
            printf("(iter=%d, θ=%d)", bumps_iter[b], bumps_theta[b]);
        }
        printf("]  total=%d\n", bump_n);
    }
#endif
```

- [ ] **Step 3: Smoke — provoke a bump on a graph that should fire**

Use a tiny rate threshold to force the controller to bump:

```bash
./CHROMA/CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/cit-Patents.egr -a cuSL_ELS_SDC_CTA_S --predict --dynamic-theta --dynamic-rate 0.001 --dynamic-K 5 2>&1 | grep -E "Dynamic θ|EGC|θ trajectory|colors used|verif"
```
Expected: `Dynamic θ:` line, `EGC θ: 3 (Predicted)` (or similar), `θ trajectory: start=3  bumps=[(iter=5, θ=4), (iter=10, θ=5), ...]`, `result verification passed`.

If no bumps fire, lower `--dynamic-rate` further or pick a graph with bigger `iter_count` (cit-Patents typically has 800+ iters).

- [ ] **Step 4: Smoke — verify no print when --dynamic-theta is off**

```bash
./CHROMA/CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "θ trajectory" || echo "(no trajectory printed — correct)"
```
Expected: `(no trajectory printed — correct)`.

- [ ] **Step 5: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "online-θ (T9): trajectory readback + stdout print"
```

---

## Task 10: JSON trajectory log to file

**Files:**
- Modify: `CHROMA/CHROMA.cu` (extends the trajectory block from Task 9)

- [ ] **Step 1: Add JSON write block**

Inside the `#ifdef DYNAMIC_THETA / if (dynamic_theta)` block from Task 9, after the stdout `printf("θ trajectory: ...")`, append:

```cpp
        if (!dynamic_log.empty()) {
            std::ofstream f(dynamic_log, std::ios::app);
            if (f.is_open()) {
                int cap = (dynamic_cap > 0) ? dynamic_cap : (fuzzy_number + 5);
                int theta_final = (bump_n > 0) ? bumps_theta[bump_n - 1] : fuzzy_number;
                f << "{\n";
                f << "  \"graph\": \"" << filename << "\",\n";
                f << "  \"theta_initial\": " << fuzzy_number << ",\n";
                f << "  \"theta_final\":   " << theta_final << ",\n";
                f << "  \"ctrl_K\": "    << dynamic_K    << ", "
                  << "\"ctrl_rate\": " << dynamic_rate << ", "
                  << "\"ctrl_step\": " << dynamic_step << ", "
                  << "\"ctrl_cap\": "  << cap          << ",\n";
                f << "  \"bumps\": [";
                for (int b = 0; b < bump_n; ++b) {
                    if (b > 0) f << ", ";
                    f << "{\"iter\":" << bumps_iter[b] << ",\"theta\":" << bumps_theta[b] << "}";
                }
                f << "]\n}\n";
                f.close();
                printf("θ trajectory JSON appended to %s\n", dynamic_log.c_str());
            } else {
                std::cerr << "Warning: could not open --dynamic-log path '"
                          << dynamic_log << "' for append.\n";
            }
        }
```

Make sure `<fstream>` is included near the top of `CHROMA.cu`:
```bash
grep -n "include <fstream>" CHROMA/CHROMA.cu || echo "(need to add #include <fstream>)"
```
If missing, add `#include <fstream>` near the other `#include`s.

- [ ] **Step 2: Smoke — write trajectory JSON**

```bash
rm -f /tmp/dyntheta_log.json
./CHROMA/CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/cit-Patents.egr -a cuSL_ELS_SDC_CTA_S --predict --dynamic-theta --dynamic-rate 0.001 --dynamic-K 5 --dynamic-log /tmp/dyntheta_log.json 2>&1 | grep "JSON appended"
echo "--- /tmp/dyntheta_log.json ---"
cat /tmp/dyntheta_log.json
echo "--- parse-check ---"
python3 -c "import json; d=json.loads(open('/tmp/dyntheta_log.json').read()); print('OK', d['graph'], 'bumps:', len(d['bumps']))"
```
Expected: `θ trajectory JSON appended to /tmp/dyntheta_log.json`; JSON parses cleanly; `bumps:` count > 0.

- [ ] **Step 3: Smoke — multiple invocations append**

```bash
./CHROMA/CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/facebook.egr -a cuSL_ELS_SDC_CTA_S --predict --dynamic-theta --dynamic-log /tmp/dyntheta_log.json 2>&1 | grep "JSON appended"
wc -l /tmp/dyntheta_log.json
```
Expected: line count grew (file now has 2 JSON objects).

- [ ] **Step 4: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "online-θ (T10): JSON trajectory log via --dynamic-log"
```

---

## Task 11: Three-way sweep script

**Files:**
- Create: `scripts/sweep_dynamic_theta.py`

- [ ] **Step 1: Write the sweep script**

```python
#!/usr/bin/env python3
"""Three-way EGR sweep comparing dynamic θ controller vs static v3.

  static_v3      : --predict
  dyn_only       : --dynamic-theta            (θ_initial = 0)
  static + dyn   : --predict --dynamic-theta  (θ_initial from v3 RF)

Per-graph: total time, color count, θ_initial, θ_final, n_bumps.
Aggregate (mean / geomean speedup, mean Δ colors, wins) on:
  - EGR overlap-with-train (11)
  - EGR holdout            ( 8)
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")
TRAJ_RE      = re.compile(r"θ trajectory: start=(-?\d+)\s+bumps=\[(.*?)\]\s+total=(\d+)")

def parse(out):
    egc, t, c, runs = (EGC_RE.search(out), AVG_TOTAL_RE.search(out),
                        AVG_COLOR_RE.search(out), RUN_RE.findall(out))
    if not (egc and t and c and runs):
        return None
    rec = {"theta_initial": int(egc.group(1)),
           "avg_color":     float(c.group(1)),
           "avg_total_ms":  float(t.group(1)),
           "n_runs":        len(runs)}
    traj = TRAJ_RE.search(out)
    if traj:
        rec["theta_initial_logged"] = int(traj.group(1))
        rec["n_bumps"]              = int(traj.group(3))
        # parse last bump for theta_final
        bumps = traj.group(2).strip()
        if bumps:
            last = bumps.split(",")[-1]
            m = re.search(r"θ=(\d+)", last)
            if m: rec["theta_final"] = int(m.group(1))
        else:
            rec["theta_final"] = rec["theta_initial_logged"]
    return rec


def run_one(binary, graph, mode, runs, timeout, dyn_K, dyn_rate, dyn_step, dyn_cap):
    cmd = [binary, "-f", str(graph), "-a", "cuSL_ELS_SDC_CTA_S", "--runs", str(runs)]
    if mode == "static_v3":
        cmd.append("--predict")
    elif mode == "dyn_only":
        cmd.extend(["--dynamic-theta",
                    "--dynamic-K", str(dyn_K),
                    "--dynamic-rate", str(dyn_rate),
                    "--dynamic-step", str(dyn_step)])
        if dyn_cap > 0:
            cmd.extend(["--dynamic-cap", str(dyn_cap)])
    elif mode == "static_dyn":
        cmd.extend(["--predict", "--dynamic-theta",
                    "--dynamic-K", str(dyn_K),
                    "--dynamic-rate", str(dyn_rate),
                    "--dynamic-step", str(dyn_step)])
        if dyn_cap > 0:
            cmd.extend(["--dynamic-cap", str(dyn_cap)])
    else:
        raise ValueError(mode)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return parse(proc.stdout)
    except subprocess.TimeoutExpired:
        return None


def gmean(xs):
    if not xs: return 0.0
    p = 1.0
    for x in xs: p *= x
    return p ** (1.0 / len(xs))


OVERLAP = {"Email-Enron.col.egr","Slashdot0811.egr","Slashdot0902.egr","Stanford.egr",
           "as-skitter.egr","cit-Patents.egr","delaunay_n24.egr","soc-Epinions1.col.egr",
           "wiki-Talk.col.egr","wiki-Vote.col.egr","youtube.egr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="CHROMA/CHROMA")
    ap.add_argument("--egr-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dyn-K", type=int, default=10)
    ap.add_argument("--dyn-rate", type=float, default=0.005)
    ap.add_argument("--dyn-step", type=int, default=1)
    ap.add_argument("--dyn-cap", type=int, default=0)
    ap.add_argument("--out", default="model/v2_data/egr_dynamic_theta.json")
    args = ap.parse_args()

    graphs = sorted(Path(args.egr_dir).glob("*.egr"))
    print(f"# {len(graphs)} graphs, runs={args.runs}, K={args.dyn_K}, "
          f"rate={args.dyn_rate}, step={args.dyn_step}, cap={args.dyn_cap or 'auto'}",
          file=sys.stderr)

    rows = []
    for g in graphs:
        rec = {"graph": g.name, "in_train_overlap": g.name in OVERLAP}
        baseline_cmd = [args.binary, "-f", str(g), "-a", "cuSL_ELS_SDC_CTA_S",
                         "--runs", str(args.runs), "-e", "0"]
        try:
            rec["baseline"] = parse(subprocess.run(baseline_cmd, capture_output=True,
                                                    text=True, timeout=args.timeout).stdout)
        except subprocess.TimeoutExpired:
            rec["baseline"] = None
        for m in ("static_v3", "dyn_only", "static_dyn"):
            rec[m] = run_one(args.binary, g, m, args.runs, args.timeout,
                              args.dyn_K, args.dyn_rate, args.dyn_step, args.dyn_cap)

        def fmt(r):
            if r is None: return "?"
            extras = ""
            if "n_bumps" in r:
                extras = f"  bumps={r['n_bumps']:>2}  θf={r.get('theta_final', '?')}"
            return f"θ={r['theta_initial']:>2} {r['avg_total_ms']:7.1f}ms {r['avg_color']:5.1f}c{extras}"

        flag = "⚠trained" if rec["in_train_overlap"] else "·holdout"
        print(f"{g.name:32s} {flag} | base {fmt(rec['baseline']):26s} | "
              f"sv3 {fmt(rec['static_v3']):26s} | "
              f"dyn {fmt(rec['dyn_only']):40s} | "
              f"sv3+dyn {fmt(rec['static_dyn']):40s}",
              file=sys.stderr)
        rows.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # Aggregate
    modes = ["static_v3", "dyn_only", "static_dyn"]
    for label, subset in [("ALL 19", rows),
                           ("OVERLAP 11", [r for r in rows if r["in_train_overlap"]]),
                           ("HOLDOUT 8", [r for r in rows if not r["in_train_overlap"]])]:
        full = [r for r in subset if r["baseline"] and all(r[m] for m in modes)]
        if not full:
            continue
        bt = [r["baseline"]["avg_total_ms"] for r in full]
        bc = [r["baseline"]["avg_color"]    for r in full]
        print(f"\n=== {label} ({len(full)}) ===", file=sys.stderr)
        print(f'{"metric":24s}  ' + "  ".join(f"{m:>14s}" for m in modes), file=sys.stderr)
        for label2, key in [("mean speedup", "spd"), ("geomean speedup", "gspd"),
                             ("mean Δ colors", "dc"), ("mean predicted θ", "th"),
                             ("wins vs base", "wins"), ("graphs ramped", "rampn")]:
            cells = []
            for m in modes:
                ts = [r[m]["avg_total_ms"] for r in full]
                cs = [r[m]["avg_color"]    for r in full]
                ths = [r[m]["theta_initial"] for r in full]
                spd = [b/x if x > 0 else 0 for b, x in zip(bt, ts)]
                rampn = sum(1 for r in full if r[m].get("n_bumps", 0) > 0)
                v = {"spd": sum(spd)/len(spd),
                     "gspd": gmean(spd),
                     "dc": sum(cs)/len(cs) - sum(bc)/len(bc),
                     "th": sum(ths)/len(ths),
                     "wins": sum(1 for s in spd if s > 1.05),
                     "rampn": rampn}[key]
                if key in ("spd","gspd"): cells.append(f"{v:.2f}×")
                elif key == "dc":         cells.append(f"{v:+.2f}")
                elif key == "th":         cells.append(f"{v:.2f}")
                elif key == "wins":       cells.append(f"{v}/{len(full)}")
                elif key == "rampn":      cells.append(f"{v}/{len(full)}")
            print(f"{label2:24s}  " + "  ".join(f"{c:>14s}" for c in cells),
                  file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and quick-check argparse**

```bash
chmod +x scripts/sweep_dynamic_theta.py
python3 scripts/sweep_dynamic_theta.py --help 2>&1 | head -15
```
Expected: argparse help printed, no errors.

- [ ] **Step 3: Single-graph dry-run on facebook**

```bash
python3 scripts/sweep_dynamic_theta.py --runs 2 \
  --egr-dir /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR \
  --binary CHROMA/CHROMA \
  --out /tmp/sweep_dyntheta_test.json 2>&1 | grep -E "facebook|HOLDOUT" | head -5
```
Expected: `facebook.egr` line shows the four modes (base, sv3, dyn, sv3+dyn), and the HOLDOUT aggregate prints.

- [ ] **Step 4: Commit**

```bash
git add scripts/sweep_dynamic_theta.py
git commit -m "online-θ (T11): three-way sweep script (static_v3 / dyn_only / static + dyn)"
```

---

## Task 12: Run baseline three-way sweep on EGR

**Files:**
- Generated: `model/v2_data/egr_dynamic_theta.json`

- [ ] **Step 1: Run the full sweep with default tunables**

```bash
cd /home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2
python3 scripts/sweep_dynamic_theta.py --runs 5 \
  --binary CHROMA/CHROMA \
  --egr-dir /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR \
  --out model/v2_data/egr_dynamic_theta.json 2>&1 | tee /tmp/sweep_dyntheta.log | tail -60
```
Expected wall time: ~5–15 minutes (19 graphs × 4 modes × 5 runs avg). Per-graph row + aggregate tables for ALL 19 / OVERLAP 11 / HOLDOUT 8.

- [ ] **Step 2: Inspect aggregates**

```bash
grep -E "^=== |mean speedup|geomean speedup|mean Δ|wins vs base|graphs ramped" /tmp/sweep_dyntheta.log
```
Read the holdout row carefully — that's the paper-honest comparison.

- [ ] **Step 3: Apply Decision Criterion (from spec)**

Eyeball the HOLDOUT table:
- Does `static_dyn` geomean ≥ `static_v3` × 1.10? (10% improvement)
- Is `static_dyn` Δcolors ≤ `static_v3` Δcolors + 0.5? (no significant color regression)

If BOTH yes → dynamic θ is a clear win; record decision in commit message.
If either fails → record as "no clear win at default tunables; hyperparameter sweep next" (Task 13).

- [ ] **Step 4: Commit data + decision summary**

```bash
git add -f model/v2_data/egr_dynamic_theta.json
git commit -m "online-θ (T12): baseline three-way sweep on EGR (defaults K=10, rate=0.005, step=1)

Results summary:
  HOLDOUT 8:
    static_v3       : geomean ?.??×, Δcolors ?.??, wins ?/8
    dyn_only        : geomean ?.??×, Δcolors ?.??, wins ?/8, ramped ?/8
    static + dyn    : geomean ?.??×, Δcolors ?.??, wins ?/8, ramped ?/8

  Decision: <ship | hyperparameter sweep | negative result>"
```
Replace the `?` placeholders with actual numbers from Step 2 output before committing.

---

## Task 13: Hyperparameter sweep (only if Task 12 didn't find a clean win)

**Files:**
- Modify (optional): `scripts/sweep_dynamic_theta.py` (or use a new wrapper)
- Generated: `model/v2_data/egr_dyntheta_hp.csv`

- [ ] **Step 1: Decide whether to run**

If Task 12 already shows `static_dyn` clearly beats `static_v3` (≥10% geomean, no color regression), **SKIP** this task — go straight to Task 14.

If results are mixed or null, run the hyperparameter sweep:

- [ ] **Step 2: Write a tiny driver that wraps Task 11's script over a grid**

```bash
cat > /tmp/dyntheta_hp.sh << 'EOF'
#!/bin/bash
set -e
out_csv=model/v2_data/egr_dyntheta_hp.csv
echo "K,rate,step,cap,subset,geomean_spd,mean_dc,wins,ramped" > $out_csv

for K in 5 10 20 50; do
for rate in 0.001 0.005 0.01 0.05; do
for step in 1 2; do
  out_json=/tmp/sweep_dyntheta_K${K}_r${rate}_s${step}.json
  python3 scripts/sweep_dynamic_theta.py --runs 3 \
    --binary CHROMA/CHROMA \
    --egr-dir /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR \
    --dyn-K $K --dyn-rate $rate --dyn-step $step \
    --out $out_json 2>&1 | tee /tmp/sweep_dyntheta_run.log >/dev/null

  # Extract HOLDOUT static_dyn aggregate from the log
  geomean=$(grep -A6 "=== HOLDOUT" /tmp/sweep_dyntheta_run.log | grep "geomean speedup" | awk '{print $NF}' | tr -d '×')
  dc=$(grep      -A6 "=== HOLDOUT" /tmp/sweep_dyntheta_run.log | grep "mean Δ colors"   | awk '{print $NF}')
  wins=$(grep    -A6 "=== HOLDOUT" /tmp/sweep_dyntheta_run.log | grep "wins vs base"    | awk '{print $NF}')
  ramped=$(grep  -A6 "=== HOLDOUT" /tmp/sweep_dyntheta_run.log | grep "graphs ramped"   | awk '{print $NF}')
  echo "$K,$rate,$step,auto,HOLDOUT,$geomean,$dc,$wins,$ramped" >> $out_csv
done; done; done

echo "Wrote $out_csv"
sort -t, -k6 -nr $out_csv | head -10
EOF
chmod +x /tmp/dyntheta_hp.sh
/tmp/dyntheta_hp.sh
```

ETA: 4 (K) × 4 (rate) × 2 (step) = 32 runs × ~3 min each ≈ 90 minutes. Run in background or leave overnight.

- [ ] **Step 3: Pick the winning combo + commit CSV**

```bash
sort -t, -k6 -nr model/v2_data/egr_dyntheta_hp.csv | head -5
```
The top row's (K, rate, step) is the deployment recommendation.

```bash
git add -f model/v2_data/egr_dyntheta_hp.csv
git commit -m "online-θ (T13): hyperparameter sweep — best (K, rate, step) on holdout"
```

- [ ] **Step 4: Re-run Task 12 with the winning combo**

```bash
python3 scripts/sweep_dynamic_theta.py --runs 5 \
  --binary CHROMA/CHROMA \
  --dyn-K <best_K> --dyn-rate <best_rate> --dyn-step <best_step> \
  --out model/v2_data/egr_dynamic_theta_tuned.json
git add -f model/v2_data/egr_dynamic_theta_tuned.json
git commit -m "online-θ (T13): re-run holdout sweep with tuned (K, rate, step)"
```

---

## Task 14: Decision and merge to main (if positive result)

**Files:** none (process step)

- [ ] **Step 1: Restate decision criterion against final numbers**

From either Task 12 or Task 13 (whichever produced the deployable result), check:

| condition | required |
|-----------|---------|
| HOLDOUT geomean(static_dyn) ≥ geomean(static_v3) × 1.10 | YES → ship |
| HOLDOUT Δcolors(static_dyn) ≤ Δcolors(static_v3) + 0.5 | YES → ship |
| Worst-case any single graph color regression > +2 | NO → ship |

If all three satisfied → proceed to Step 2.
If not → stop here. Branch stays as ablation; don't merge to main. Commit a "negative result" note.

- [ ] **Step 2: Make `--dynamic-theta` the default (when DYNAMIC_THETA built)**

In `CHROMA/CHROMA.cu`, change the default of `bool dynamic_theta = false;` to `true`. Add an `--no-dynamic-theta` opt-out flag for ablation:

```cpp
} else if (strcmp(argv[i], "--no-dynamic-theta") == 0) {
    dynamic_theta = false;
```

Update help text accordingly.

- [ ] **Step 3: Update CHROMA Makefile to make DYNAMIC_THETA=1 the default**

In `CHROMA/Makefile`, change:
```makefile
DYNAMIC_THETA ?= 0
```
to:
```makefile
DYNAMIC_THETA ?= 1
```

- [ ] **Step 4: Re-run smoke + commit**

```bash
cd CHROMA && make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -1
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/europe_osm.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "Dynamic θ|EGC|θ trajectory|colors used|verif"
cd ..
git add CHROMA/CHROMA.cu CHROMA/Makefile
git commit -m "online-θ (T14): promote dynamic θ to default; --no-dynamic-theta opt-out"
```

- [ ] **Step 5: Merge `theta-predictor-online` into main**

```bash
cd /home/chsieh45/PunchShadow/CHROMA
git status --short | head
git merge --no-ff theta-predictor-online -m "Merge branch 'theta-predictor-online': online dynamic θ controller

On-device controller monotonically raises FuzzyNumber when removal rate
indicates the residual graph is in active-peeling phase. Layered on
static v3 predictor: v3 picks θ_initial, controller ramps from there.

Default: ON (build with PRE_MODEL=1; opt-out with --no-dynamic-theta).

Holdout 8 (paper-honest):
  static_v3      : geomean ?.??×, Δcolors ?.??, wins ?/8
  static + dyn   : geomean ?.??×, Δcolors ?.??, wins ?/8, ramped ?/8

Spec: docs/superpowers/specs/2026-05-11-online-theta-prediction-design.md
Plan: docs/superpowers/plans/2026-05-11-online-theta-prediction.md"
```

Replace `?` with the final numbers before committing.

- [ ] **Step 6: Verify main HEAD + sanity smoke**

```bash
cd /home/chsieh45/PunchShadow/CHROMA/CHROMA
make clean > /dev/null 2>&1 && make ARCH=sm_86 PRE_MODEL=1 2>&1 | tail -1
./CHROMA -f /home/chsieh45/PunchShadow/CHROMA/Datasets/EGR/europe_osm.egr -a cuSL_ELS_SDC_CTA_S --predict 2>&1 | grep -E "Dynamic θ|EGC|θ trajectory|colors used|verif"
cd ..
git log --oneline -5
```
Expected: dynamic θ active, europe_osm verification passes, main HEAD is the merge commit.

---

## Self-Review

**Spec coverage:**

| spec section | implementing task(s) |
|-------------|---------------------|
| Goal: monotone non-decreasing θ via on-device controller | T1 (state), T2 (params), T6 (controller block) |
| Architecture: single-thread block-0 controller at iteration boundary | T6 (placement inside existing block) |
| Controller logic (concrete pseudocode) | T6 (matches spec verbatim) |
| State variables (`g_nodes`, `last_remove_size`, `bump_*`, `CTRL_*`) | T1 |
| `setDynamicParameters` host launcher | T2 |
| `resetForRun()` extension | T3 |
| 5 new CLI flags (`--dynamic-theta` + 4 tunables + log) | T7, T8 (wiring), T10 (log) |
| Compile-time gate `-DDYNAMIC_THETA` | T4 (Makefile), T1/T2/T3/T6/T8/T9/T10 (#ifdef wrap) |
| Trajectory `(iter, new_θ)` log | T1 (state), T6 (writer), T9 (readback + stdout) |
| JSON trajectory output | T10 |
| Failure modes (`delta < 0`, BUMP_LOG_MAX, no-DYNAMIC_THETA build) | T6 (clamp + cap), T8 (warn) |
| Three-way evaluation (static_v3 / dyn_only / static + dyn) | T11 (script), T12 (run) |
| Aggregate metrics + holdout split | T11 (sweep_dynamic_theta.py main()) |
| Hyperparameter sweep | T13 |
| Decision criterion | T14 (Step 1) |
| `--no-dynamic-theta` opt-out (post-promotion) | T14 (Step 2) |

**Placeholder scan:** all code blocks contain real, runnable code. No "TBD" / "fill in" / "similar to". Numbers in commit-message templates have `?` placeholders that must be replaced with actual results before committing — this is intentional and called out in each step.

**Type consistency:**
- Device variables (`__device__ int CTRL_K`, etc.) declared in T1, used by name in T2, T6.
- Host wrapper `setDynamicParameters(int, int, float, int, int)` signature: T2 declares, T8 calls — matched.
- CLI variable names (`dynamic_K`, `dynamic_rate`, etc.) consistent across T7, T8, T9, T10.
- `bump_count` / `bump_iter[]` / `bump_theta[]` consistent across T1 (declaration), T6 (writer), T9 (readback).

**Pre-existing-symbols I'm depending on:**
- `worker`, `iter_count`, `FuzzyNumber`, `g_minDegree` (existing in PA.cu) — used by controller
- `g.nodes` (host-side loaded ECLgraph) — used by T8 to pass into setDynamicParameters
- `cudaMemcpyFromSymbol`, `cudaMemcpyToSymbol`, `CUDA_CHECK` — already used elsewhere in chroma_utils.cu / CHROMA.cu
- `cooperative_groups::grid_group` (`grid.thread_rank()`) — already in P_SL_ELS_SDC_CTA_S
