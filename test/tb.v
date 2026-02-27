`timescale 1ns/1ps
// Minimal cocotb-friendly wrapper TB for Tiny Tapeout projects.
// cocotb drives clk/rst_n/ena/ui_in/uio_in directly.
module tb;
  reg        clk;
  reg        rst_n;
  reg        ena;

  reg [7:0]  ui_in;
  wire [7:0] uo_out;

  reg [7:0]  uio_in;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  // Instantiate your Tiny Tapeout top module
  tt_um_barilwil_microcore16 dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .ena    (ena),
    .ui_in  (ui_in),
    .uo_out (uo_out),
    .uio_in (uio_in),
    .uio_out(uio_out),
    .uio_oe (uio_oe)
  );
endmodule
