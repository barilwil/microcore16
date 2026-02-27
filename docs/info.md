<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project is a small RV32E (16-register) RISC‑V CPU designed for Tiny Tapeout. The chip uses Tiny Tapeout I/O
as a 16‑bit time‑multiplexed external bus:

- Address / write data out: `{uio_out[7:0], uo_out[7:0]}` (16-bit word)
- Read data in: `{uio_in[7:0], ui_in[7:0]}` (16-bit word)
- Direction control: `uio_oe = 8'hFF` when the CPU drives `uio_out` (ADDR / write DATA phases), and `uio_oe = 8'h00`
  when the CPU is reading (read DATA phase).

The CPU fetches each 32‑bit instruction as two 16‑bit reads (low halfword then high halfword). A multi‑cycle control
FSM decodes and executes the instruction. Loads/stores use the same 16‑bit bus; byte stores are implemented as a
read‑modify‑write since there are no byte enables. Shifts are implemented with a serial (multi‑cycle) shifter to save area.

## How to test

### Simulation (recommended)

1. Run the template cocotb sanity test with Verilator:
   - `cd test`
   - `make SIM=verilator`

2. Run the full self‑checking SystemVerilog testbench (directed edge cases + randomized ALU smoke):
   - Compile and run `test/tb_microcore16.sv` together with `src/processor.sv` and `src/tt_wrapper.sv`.
   - The testbench uses `test/external_memory.sv` as a simulation-only memory model and reports PASS/FAIL in the console.

### On real Tiny Tapeout hardware

This design expects an external device (FPGA or microcontroller) to emulate memory on the 16‑bit bus:

- Cycle A (ADDR): CPU drives address on `{uio_out, uo_out}` with `uio_oe=0xFF`
- Cycle D (DATA):
  - READ: external drives `{uio_in, ui_in}` while CPU sets `uio_oe=0x00`
  - WRITE: CPU drives write data on `{uio_out, uo_out}` with `uio_oe=0xFF`

## External hardware

None required for simulation. For real hardware testing, an external controller (FPGA/microcontroller) is required to
provide instruction/data memory responses on the 16‑bit bus.
