// =============================================================================
// Tiny Tapeout top-level wrapper
// - This is the module you tape out (must match the Tiny Tapeout template ports)
// - Instantiates your CPU core in processor_fixed.sv (microcore16_top)
//
// IMPORTANT:
// 1) Rename the module to match your Tiny Tapeout project name, e.g.:
//      module tt_um_wilmer_microcore16 ( ... );
// 2) Ensure this file is included in your TT repo build along with processor_fixed.sv
// =============================================================================
`timescale 1ns/1ps

module tt_um_barilwil_microcore16 (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        ena,

  input  wire [7:0]  ui_in,
  output wire [7:0]  uo_out,

  input  wire [7:0]  uio_in,
  output wire [7:0]  uio_out,
  output wire [7:0]  uio_oe
);

  // If you want the core to halt when not enabled, you can gate the clock or
  // add an enable input to microcore16_top. For now we just ignore `ena`,
  // which is acceptable in Tiny Tapeout projects.
  wire core_clk = clk;

  microcore16_top u_core (
    .clk    (core_clk),
    .rst_n  (rst_n),

    .ui_in  (ui_in),
    .uo_out (uo_out),
    .uio_in (uio_in),
    .uio_out(uio_out),
    .uio_oe (uio_oe)
  );

endmodule
