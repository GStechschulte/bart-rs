# Codex prompt: Option A (Tree Dump for `all_trees` compatibility)

You are working in the `bart-rs` repo, which aims to be a drop-in replacement for the Python package `pymc_bart`, with the sampler implemented in Rust. Currently, Python utilities like PDP plots and variable inclusion break because `_sample_posterior` expects `bartrv.owner.op.all_trees`, but the Rust sampler does not provide it.

Goal: Implement “Option A”: after sampling, expose a compact, serializable representation of the sampled trees (“tree dump”) to Python, and set it into `bartrv.owner.op.all_trees` (or whatever field `_sample_posterior` reads), so downstream PDP and variable inclusion code paths work without changing their logic.

Constraints:
- Do NOT return Python Tree objects from Rust.
- Return a lightweight tree dump per draw and per tree. Python may wrap it, but should not require full class graphs.
- Keep changes minimal and reviewable; do not refactor unrelated code.
- Avoid checking in profiling artifacts, lockfile churn, or formatting-only changes unless required to compile.

Tasks:
1) Locate the Python code that currently breaks:
   - Find `_sample_posterior` and identify exactly where it reads `bartrv.owner.op.all_trees` (or similar).
   - Determine the expected structure: dimensions and access patterns (e.g., `[chain][draw][tree]`, or `[draw][tree]`, etc.).
   - Add a brief comment in code describing the required contract.

2) Define a compact tree dump schema to transfer from Rust to Python.
   Implement a Rust struct (serde-friendly, PyO3-friendly) named `TreeDump` (or similar) that contains enough information to:
   - Evaluate the tree on arbitrary X later (for PDP).
   - Count split feature usage (for variable inclusion).
   Recommended fields (choose the minimum that matches your Tree implementation):
   - `split_feature: Vec<i32>` (len = `MAX_NODES`; `-1` indicates leaf)
   - `split_value: Vec<f64>` (len = `MAX_NODES`)
   - `left_child: Vec<i32>` (len = `MAX_NODES`; `-1` if none)
   - `right_child: Vec<i32>` (len = `MAX_NODES`; `-1` if none)
   - `leaf_value: Vec<f64>` (len = `MAX_NODES`; value used when node is leaf)
   - (optional) `root_index: i32` (usually 0)
   Keep it compact and deterministic.

3) Add a Rust function that returns posterior tree dumps.
   - Identify where sampling happens (the Rust sampler entry point called from Python).
   - Modify it so it can return (in addition to other outputs) a nested list of `TreeDump` objects for each draw and each tree.
   - The returned shape should match what Python expects in step (1).
   - Use PyO3 bindings to return `Vec<Vec<TreeDump>>` (or `Vec<Vec<Vec<TreeDump>>>` if chains are included).
   - Ensure `TreeDump` is convertible to Python types (either as a `#[pyclass]` with `Vec` fields, or as a dict/list of primitives).

4) Wire it into Python and restore compatibility:
   - In the Python wrapper code, call the Rust sampler method that returns tree dumps.
   - Assign the returned nested structure into `bartrv.owner.op.all_trees` (or the exact attribute used by `_sample_posterior`).
   - Ensure PDP and variable inclusion code can iterate/read `split_feature`/thresholds without crashing.
   - Do not redesign Python APIs; keep the same external behavior.

5) Add tests (minimum required):
   - Add a Python test that constructs a minimal model, runs `_sample_posterior` (or the relevant path), and asserts:
     a) `bartrv.owner.op.all_trees` exists after sampling
     b) it has the expected nested shape (e.g., len(draws) and len(trees))
     c) each `TreeDump` has the required arrays with consistent lengths
   - Add a test for variable inclusion that computes inclusion counts from the tree dumps and returns an array of length `n_features` (can be simple sanity checks, not exact values).

Acceptance criteria:
- Running the PDP/variable inclusion code path no longer fails due to missing `all_trees`.
- `all_trees` is populated with compact tree dumps (not Python tree objects).
- Tests pass.
- No unrelated refactors.

Implementation guidance:
- Prefer returning `TreeDump` as plain Python dicts/lists if PyO3 class plumbing is annoying.
- If `MAX_NODES` is const-generic, you can still emit `Vec`s of length `MAX_NODES`.
- If your internal `Tree` stores nodes sparsely, still emit fixed-length arrays with sentinel values.

Start by inspecting the expected `all_trees` contract in Python (`_sample_posterior`) before implementing the Rust output.
