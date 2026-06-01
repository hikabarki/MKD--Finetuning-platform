# DECISIONS

Append-only Architectural Decision Records. **Never edit past entries** — write
a new ADR that supersedes them.

---

## ADR-001 — CLI-first, guided workflow over web UI
**Date:** 2026-05-26
**Author:** Hika
**Status:** Accepted

### Context
User requested a fine-tuning platform usable by non-engineers, where each step
prints the next command. This biases toward a turnkey GUI, but a GUI is:
- 10× the engineering cost of a CLI
- Harder to reproduce / version-control
- A bottleneck for the actual training compute (the hard part is GPU orchestration, not buttons)

### Decision
**Ship a guided CLI first** (`ft add-dataset`, `ft clean-dataset`, …). Each
script ends by printing the next command verbatim. A web UI is deferred to v2;
when built, it will wrap the same CLI scripts (subprocess) so the CLI remains
the source of truth.

### Consequences
- ✅ Faster to ship; easier to debug
- ✅ Non-engineers follow a printed runbook instead of clicking
- ✅ Power users can scriptchain the same commands in CI
- ❌ Less "magical" than a web app — mitigated with very polished printed output (colors, boxes, copy-paste-ready commands)
- ❌ Requires a terminal — acceptable for the MKD-internal audience

---

## ADR-002 — Defer framework choice until Phase 2 hardware answers
**Date:** 2026-05-26
**Author:** Hika
**Status:** Accepted

### Context
Three serious candidates for the training backend:
1. **Axolotl** — most mature, YAML-driven, supports SFT/DPO/ORPO, FSDP+DeepSpeed
2. **LLaMA-Factory** — best non-engineer UX (has its own webui), broad model support
3. **Unsloth** — fastest single-GPU LoRA/QLoRA, limited to ≤2 GPUs (free tier)
4. **TRL (raw)** — most flexible, most code to write

### Decision
**Decide after intake answers `Q-001` (hardware) and `Q-002` (model size).**
Provisional bias: **Axolotl as primary**, **Unsloth as fast-path** for single-
consumer-GPU LoRA jobs. LLaMA-Factory considered if non-engineer UX is the
dominant constraint.

### Consequences
- Locks the schema for `configs/*.yaml` to Axolotl's shape (which is similar to
  HF TrainingArguments) so a future framework swap is mostly a renaming exercise.
- A formal `ADR-003 — Training backend selection` will be written after Phase 1
  answers arrive.
