# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

@cocotb.test()
async def test_sanity(dut):
    dut._log.info("Start sanity test")

    # 50 MHz-ish clock (20 ns period)
    clock = Clock(dut.clk, 20, unit="ns")
    cocotb.start_soon(clock.start())

    # Drive stable inputs
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0

    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Run some cycles and check outputs are resolvable + uio_oe is either 0x00 or 0xFF
    for _ in range(200):
        await ClockCycles(dut.clk, 1)

        assert dut.uo_out.value.is_resolvable, "uo_out is X/Z"
        assert dut.uio_out.value.is_resolvable, "uio_out is X/Z"
        assert dut.uio_oe.value.is_resolvable, "uio_oe is X/Z"

        oe = int(dut.uio_oe.value)
        assert oe in (0x00, 0xFF), f"uio_oe unexpected: 0x{oe:02x}"

    dut._log.info("Sanity test passed")
