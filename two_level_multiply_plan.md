# Two-level compressed multiply: implementation plan

Design: `~/src/madness/madness/patches-2026-11-16/two_level_multiply.md`. Current version:
`include/mra/tasks/multiply_compressed.h`, `include/mra/kernels/multiply_compressed.h`.

## Findings that shape the plan

1. **PaRSEC parameter limit.** This repo's `build/` has `PARSEC_MAX_PARAM_COUNT=20`, MADNESS's
   build 32. Today's task uses up to 18 buffers. The two-level task needs 32: reconfigure with
   `-DPARSEC_MAX_PARAM_COUNT=32` before running it here.
2. **No extra scratch buffer.** The children's scaling coefficients between the two steps live in
   scratch slots of the packed output (a `CombinedReconstructedNode`, tag `MULC2_SCRATCH`): the
   kernel addresses them through sparse views, which need device sparsity bytes, and `out_arr`
   must hold views of one type. Empty nodes add no buffer. Worst case 29 of 32 buffers (db, 4
   constants, tmp, work, a, b, 16 children, s, result, packed, down).
3. **Both steps reuse `multiply_compressed_process_one` unchanged.**
   - Step 1: today's call at K; the `MULC_PASS` targets point into scratch.
   - Step 2: per continuing child c, `s_in` = scratch slice c, `a_node`/`b_node` = a(c)/b(c),
     `out_arr` = the grandchildren's targets (packed leaf slot or supernode slot).
   - The planner: `plan_multiply_compressed` at K, then at each c with the step-1 output as `s_in`.
4. **Two launches first.** With NA=1 only one work item writes a's s for a child (`write_a`) and
   other blocks read it in step 2. A grid boundary (two launches on one stream) handles this and
   gives step 2 parallelism per (function, child).
5. **Balance broadcasts.** Empty nodes for a missing a(K) go to up to 64 grandchildren x 9
   terminals x 2 factors: one broadcast per factor of one empty node over all its terminals
   (as `derivative.h`), not one send each.

## CombinedReconstructedNode

Several reconstructed nodes at different keys in one buffer, one data copy. Used for the
grandchildren's scaling coefficients (one broadcast) and for the packed product outputs (child
products and grandchild leaves). Not specific to multiply, so it lives in `include/mra/tensor/`.

- Each slot has the layout of a `SparseTensorView` storage: per-function `SparsityState` bytes,
  padded (`align_to_type`), then the allocated functions' data, packed. A slot's view is
  `SparseTensorView<T,NDIM+1>(ptr + offset[g], {M_g, K, ..., K})`; kernels need no change.
- Slots are sized to their allocated functions (the plan knows the sparsity before allocating),
  offsets rounded up to `MRA_BUFFER_ALIGNMENT`.
- Host metadata per slot, serialized: key, offset, `SparsityInfo`, leaf statuses.
- Device sparsity bytes: per slot, an `extra_dsts` entry in the batch's grouped copy (batched), or
  a per-slot push (unbatched). Never fill on the host and `SyncIn` the whole buffer.
- `slot_view(g)` builds the view from the buffer pointer; no tensor-level `populate_host_sparsity`.
- Unpacking a slot into a `FunctionsReconstructedNode` is one contiguous D2D copy (same header
  size, data packed at the front) plus `apply_sparsity` and leaf statuses on the host.
- Root: a one-slot node, every function `Inner`, none allocated.

## Status (2026-11-17)

- Done, host build, `mra-mad-function-ops-host-parsec` passes with 1 and 2 levels (elementwise,
  broadcast; tasks at levels 0-6, trees ending at odd and even levels, balance in both factors):
  - phase 0 checks;
  - phase 1: `plan_multiply_compressed_layout` (metadata only) + `plan_multiply_compressed`;
  - `include/mra/tensor/combinednode.h`;
  - `make_multiply_compressed_2level`, unbatched: step 1 and step 2 are launches of the existing
    `submit_multiply_compressed_kernel` (no kernel changes); `set_multiply_compressed_levels`,
    `MRA_TTG_MULTIPLY_LEVELS`; `make_multiply_compressed` dispatches for NDIM=3.
- Batching: the task builds a list of `MulCompressedLaunch` (step 1, then one per child with
  work items), launched one by one unbatched, or passed to `coop()` in a
  `MulCompressed2BatchMember` with the sparsity bytes of its outputs (result, every slot of packed
  and down). `submit_multiply_compressed_2level_batch_leader` stages everything with one grouped
  copy (args of both steps, two offset arrays, sparsity, work items) and launches the existing
  `multiply_compressed_kernel_batched` twice. Batched, step 2's launches run concurrently, so tmp
  holds max(step-1 items, sum of step-2 items) and each child has its own part; no work buffer
  (the leader stages the items). Shares the one-level TT's batch pools.
- CUDA build (Docker, nvcc 12.8 / gcc 12): `multiply_compressed.cu` (kernels, explicit
  instantiations) compiles; the TT code compiles with g++ in CUDA mode (MRA_ENABLE_CUDA, explicit
  instantiation), batched and unbatched, with and without proc/device maps. Through nvcc, cicc
  needs more than 7.5 GB for the two-level TT (2.9 GB for the one-level square): compile TT code
  with the host compiler, or give nvcc more memory.
- Scratch split: the scaling coefficients between the two steps are in a third combined node
  that only the task holds (`scratch`), so packed holds only product nodes (worst case 30 of 32
  buffers). The slot tags are gone.
- Compress without unpacking (`include/mra/tasks/compress_combined.h`):
  - `make_multiply_compressed_combined`: `out` gets only the product nodes at even levels,
    `packed` the packed output keyed by the task's key; no emit task.
  - `make_compress_combined` (same parameters as `make_compress`, plus the packed edge):
    - router on the even-level nodes: inner ones to compress-even, leaves up as p with an empty
      kept (a guard: a multiply task's node always has a function that continues);
    - dispatch on packed: packed to compress-even at K and (one broadcast) to compress-odd at the
      children with a continuing function; for every leaf slot an empty p and kept on its
      parent's terminals;
    - compress-even (node from the router) and compress-odd (own node = slot in the parent's
      packed) share one task body: a child with a leaf slot in packed is read from the slot,
      otherwise from the terminal. The kernels, the coop arguments and the batch leader are those
      of `make_compress`; both TTs join `batch_group`.
    - compress-odd is mapped by the parent's key (packed stays local).
  - Host tests pass (elementwise and broadcast against MADNESS, well-formed), with tasks at all
    levels, 0-8 children from slots, placeholders to both TTs. CUDA-mode g++ compile passes.
- Not done:
  - emit copies are one `memcpy_async` per node, not batched (only when unpacking);
  - functional runs on a GPU (water10/20, nsys);
  - two-level compress (a gather of up to 64 children).

## Phases

### 0. Check assumptions
- Compress output: functions absent at K have every child flag set to leaf (routers depend on it).
- A multi-terminal `ttg::device::broadcast` holds one data copy.
- Invariant: a node a(K) exists => the product has a task at K (else an input dangles).

### 1. Refactor, no behavior change
Split `plan_multiply_compressed` into a metadata-only planner (statuses, sparsity, task and
node-at flags, work items) and allocation by the caller. Test with `mad_function_ops_test`.

### 2. CombinedReconstructedNode
The type, its serialization, `slot_view`, the device sparsity push, the slot → node unpack.

### 3. Kernels
- Unbatched: no new kernel. Step 1 is `submit_multiply_compressed_kernel` at K, step 2 one launch
  per continuing child c (key c, a(c), b(c), s = scratch slot c, outputs in packed/down slots).
- Batched: the same batched kernel, launched twice: members = tasks (step 1), then members =
  (task, child) pairs (step 2); one grouped copy with both argument lists and the slot sparsity
  bytes as extra entries; tmp per task sized for max(step-1 items, sum of step-2 items).
- CUDA build in Docker (CLAUDE.md).

### 4. Task graph: `make_multiply_compressed_2level`, same signature

| TT | key | inputs → outputs |
|---|---|---|
| route_a / route_b (Space, no kernel) | node key | even K: to the task's A, one broadcast of an empty node to A_c where a has no node. Odd c: to (parent(c), A_ci(c)) |
| multiply2 | even K | a, b, 8 a_c, 8 b_c, supernode → result at K on `out`, packed → emit, supernode → grandchildren with a task, mask → balance |
| balance2 | K | per factor one broadcast of an empty node to the A/A_c terminals of grandchildren that need it |
| emit | child c | packed → up to 9 nodes on `out` |
| primer | root | one-slot supernode |

Routers and emits mapped by their multiply task's key (`parent(c)`).

### 5. Bring-up order
Host path (`MRA_ENABLE_HOST`, no buffer limit) → device unbatched → batched.

### 6. Switch and tests
- `MRA_TTG_MULTIPLY_LEVELS`, read once; `make_multiply_compressed` dispatches, MADNESS unchanged.
- `test_multiply_compressed` runs both: elementwise, broadcast, depths 1/2/3, a factor ending at an
  odd level, node-by-node comparison with the one-level output.
- Water10/water20 energies and the nsys scripts (kernel count, idle after multiply, batch sizes).

### 7. Later
Fused single-block kernel (step 2 reads ua/ub from its own tmp); square; mapping grandchildren by
the grandparent's device; packing the constant buffers.

## Open questions, recommendations
Q1 emit tasks · Q3 scratch in tmp (30/32) · Q4 tmp grows by ~8·M·K^3 per task, check the pool
share at batch 128 · Q5 two launches · Q6 test depth 1 · Q7 square after multiply · Q8 env switch.
