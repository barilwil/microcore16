// =============================================================================
// Tiny Tapeout External 16-bit Memory/Controller Model
// Compatible with mem_if_adapter_tt "time-mux" 2-cycle protocol:
//
//  Cycle 0 (ADDR): DUT drives address on {uio_out, uo_out} with uio_oe=8'hFF
//  Cycle 1 (DATA):
//    - READ: DUT releases uio (uio_oe=8'h00); external drives {uio_in, ui_in}
//    - WRITE: DUT drives write data on {uio_out, uo_out} with uio_oe=8'hFF;
//             external latches it and writes to memory.
//
// Notes:
// - This is a *simulation-friendly* external memory model. For real silicon,
//   you'd replace it with an actual memory interface circuit.
// - Memory is modeled as 16-bit halfwords addressed by byte address[15:0].
//   We map byte address -> halfword index via (addr >> 1).
// =============================================================================

`timescale 1ns/1ps

module tt_external_memory #(
  parameter int ADDR_BITS   = 16,            // byte address width
  parameter int WORD_BITS   = 16,            // 16-bit external bus
  parameter int WORDS       = 32768,         // 64KB / 2 bytes per halfword
  parameter bit STRICT_ALIGN = 1'b0,         // if 1, flag odd addresses
  parameter string INIT_HEX = ""             // optional $readmemh file of 16-bit words
)(
  input  logic               clk,
  input  logic               rst_n,

  // Signals from DUT (TinyTapeout-style pins)
  input  logic [7:0]         uo_out,          // DUT out [7:0] (addr/data low)
  input  logic [7:0]         uio_out,         // DUT out [15:8] (addr/data high when uio_oe=FF)
  input  logic [7:0]         uio_oe,          // 8'hFF => DUT driving upper byte, 8'h00 => DUT reading upper byte

  // Signals driven *to* DUT (as if external world is driving ui_in/uio_in)
  output logic [7:0]         ui_in_drive,     // drives DUT ui_in  [7:0]
  output logic [7:0]         uio_in_drive     // drives DUT uio_in [15:8]
);

  // -----------------------------------------------------------------------------
  // Memory storage: 16-bit halfwords
  // -----------------------------------------------------------------------------
  logic [WORD_BITS-1:0] mem [0:WORDS-1];

  initial begin
    integer i;
    for (i = 0; i < WORDS; i++) mem[i] = '0;

    if (INIT_HEX != "") begin
      $display("[tt_external_memory] Loading INIT_HEX=%0s", INIT_HEX);
      $readmemh(INIT_HEX, mem);
    end
  end

  // -----------------------------------------------------------------------------
  // Protocol tracking
  // -----------------------------------------------------------------------------
  typedef enum logic [0:0] { PH_IDLE = 1'b0, PH_DATA = 1'b1 } phase_t;
  phase_t phase_q, phase_d;

  logic [ADDR_BITS-1:0] addr_q, addr_d;

  // Convenience
  logic [15:0] dut_word16;
  assign dut_word16 = {uio_out, uo_out};

  // Byte address -> halfword index
  logic [$clog2(WORDS)-1:0] hword_idx;
  always_comb begin
    hword_idx = dut_word16[ADDR_BITS-1:1]; // addr >> 1
  end

  // -----------------------------------------------------------------------------
  // Drive defaults (external bus driven only on read-data phase)
  // -----------------------------------------------------------------------------
  always_comb begin
    ui_in_drive  = 8'h00;
    uio_in_drive = 8'h00;

    if (phase_q == PH_DATA) begin
      // READ if DUT released uio (and thus expects external drive)
      if (uio_oe == 8'h00) begin
                logic [15:0] rword;
        logic [$clog2(WORDS)-1:0] ridx;
        ridx = addr_q[ADDR_BITS-1:1]; // use latched address from ADDR phase
        if (ridx < WORDS) rword = mem[ridx];
        else              rword = 16'h0000;
ui_in_drive  = rword[7:0];
        uio_in_drive = rword[15:8];
      end
    end
  end

  // -----------------------------------------------------------------------------
  // Sequential protocol + memory write
  // -----------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase_q <= PH_IDLE;
      addr_q  <= '0;
    end else begin
      phase_q <= phase_d;
      addr_q  <= addr_d;

      // Perform write in DATA phase when DUT is driving data
      if (phase_q == PH_DATA && uio_oe == 8'hFF) begin
        // write data is on dut_word16 this cycle; address latched from previous ADDR phase
        logic [$clog2(WORDS)-1:0] widx;
        widx = addr_q[ADDR_BITS-1:1]; // addr_q >> 1

        if (STRICT_ALIGN && addr_q[0]) begin
          $display("[tt_external_memory] WARNING: odd byte address write @%0t addr=0x%0h", $time, addr_q);
        end

        if (widx < WORDS) begin
          mem[widx] <= dut_word16;
        end
      end
    end
  end

  // -----------------------------------------------------------------------------
  // Next-state: latch address on ADDR phase, move to DATA, then back to IDLE
  // -----------------------------------------------------------------------------
  always_comb begin
    phase_d = phase_q;
    addr_d  = addr_q;

    unique case (phase_q)
      PH_IDLE: begin
        // Detect ADDR phase: DUT driving uio and output word is address
        if (uio_oe == 8'hFF) begin
          addr_d  = dut_word16;
          phase_d = PH_DATA;
        end
      end

      PH_DATA: begin
        // DATA phase consumes exactly one cycle. After it, return to idle.
        // (If DUT starts a new transaction immediately, next cycle will re-latch.)
        phase_d = PH_IDLE;
      end

      default: begin
        phase_d = PH_IDLE;
      end
    endcase
  end

endmodule
