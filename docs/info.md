<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

Explain how your project works

## How to test

Explain how to use your project

## External hardware

List external hardware used in your project (e.g. PMOD, LED display, etc), if any


# Setup Notes for MicroCore-16 Tiny Tapeout Project

## What goes on the chip
Only the files in `src/` are synthesized and taped out:
- `src/tt_wrapper.sv` (Tiny Tapeout top-level module)
- `src/processor.sv` (CPU core)

## What is simulation-only
The `test/` folder contains simulation-only artifacts:
- cocotb sanity test (`test/test.py`, `test/tb.v`)
- optional heavier SV testbenches (`test/tb_microcore16.sv`, `test/external_memory.sv`)

## After you paste your final working RTL
1. Ensure the top module name in `src/tt_wrapper.sv` matches `project.top_module` in `info.yaml`.
2. Ensure `source_files` in `info.yaml` lists `processor.sv` and `tt_wrapper.sv`.
3. Run `make -B` inside `test/` (cocotb sanity).
