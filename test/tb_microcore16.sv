`timescale 1ns/1ps
// =============================================================================
// Self-checking testbench for TinyTapeout MicroCore-16 (RV32E split-fetch / 16-bit bus)
// - Instantiates tt_wrapper.sv (tt_um_wilmer_microcore16) + external_memory.sv (tt_external_memory)
// - Loads multiple directed programs into external memory model
// - Checks edge cases: sign/zero extension, SB read-modify-write, branches signed/unsigned,
//   serial shifts, x0 hardwire, bus protocol correctness.
// - Reports basic metrics (cycles, retired instructions, mem ops, shift-step cycles)
//
// Compile (Icarus Verilog):
//   iverilog -g2012 -o sim.out tb_microcore16.sv tt_wrapper.sv processor.sv external_memory.sv
// Run:
//   vvp sim.out
// Waveforms:
//   gtkwave waves.vcd
//
// If you renamed the TT top module, update DUT_MODULE below.
// =============================================================================

module tb_microcore16;

  // ----------------------------
  // Clock / reset
  // ----------------------------
  localparam time CLK_PERIOD = 10ns;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic ena   = 1'b1;

  always #(CLK_PERIOD/2) clk = ~clk;

  // ----------------------------
  // Tiny Tapeout pin bundle
  // ----------------------------
  wire [7:0] ui_in;
  wire [7:0] uio_in;

  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  // ----------------------------
  // External memory model drives ui_in/uio_in
  // ----------------------------
  wire [7:0] ui_in_drive;
  wire [7:0] uio_in_drive;

  assign ui_in  = ui_in_drive;
  assign uio_in = uio_in_drive;

  // ----------------------------
  // DUT + Memory
  // ----------------------------
  // Update this if you renamed your TT wrapper module.
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

  tt_external_memory u_mem (
    .clk         (clk),
    .rst_n       (rst_n),
    .uo_out      (uo_out),
    .uio_out     (uio_out),
    .uio_oe      (uio_oe),
    .ui_in_drive (ui_in_drive),
    .uio_in_drive(uio_in_drive)
  );

  // =============================================================================
  // RISC-V encoders (RV32I/RV32E subset)
  // =============================================================================
  function automatic [31:0] enc_R(
    input int opcode,
    input int funct3,
    input int funct7,
    input int rd,
    input int rs1,
    input int rs2
  );
    enc_R = {funct7[6:0], rs2[4:0], rs1[4:0], funct3[2:0], rd[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_I(
    input int opcode,
    input int funct3,
    input int rd,
    input int rs1,
    input int   imm12
  );
    enc_I = {imm12[11:0], rs1[4:0], funct3[2:0], rd[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_I_sh(
    input int opcode,
    input int funct3,
    input int funct7,
    input int rd,
    input int rs1,
    input int shamt
  );
    enc_I_sh = {funct7[6:0], shamt[4:0], rs1[4:0], funct3[2:0], rd[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_S(
    input int opcode,
    input int funct3,
    input int rs1,
    input int rs2,
    input int   imm12
  );
    enc_S = {imm12[11:5], rs2[4:0], rs1[4:0], funct3[2:0], imm12[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_B(
    input int opcode,
    input int funct3,
    input int rs1,
    input int rs2,
    input int   imm13   // must be multiple of 2
  );
    // imm[12|10:5|4:1|11] per RISC-V spec
    enc_B = {imm13[12], imm13[10:5], rs2[4:0], rs1[4:0], funct3[2:0], imm13[4:1], imm13[11], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_U(
    input int opcode,
    input int rd,
    input int imm20
  );
    enc_U = {imm20[19:0], rd[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] enc_J(
    input int opcode,
    input int rd,
    input int   imm21   // must be multiple of 2
  );
    // imm[20|10:1|11|19:12]
    enc_J = {imm21[20], imm21[10:1], imm21[11], imm21[19:12], rd[4:0], opcode[6:0]};
  endfunction

  function automatic [31:0] NOP();
    NOP = enc_I(7'h13, 3'h0, 0, 0, 0); // addi x0,x0,0
  endfunction

  // =============================================================================
  // Helpers to poke and observe DUT state (hierarchical)
  // =============================================================================
  function automatic [31:0] get_reg32(input int r);
    logic [15:0] lo, hi;
    begin
      lo = dut.u_core.u_rf.rf_lo[r[3:0]];
      hi = dut.u_core.u_rf.rf_hi[r[3:0]];
      get_reg32 = {hi, lo};
    end
  endfunction

  task automatic check_reg32(input int r, input [31:0] exp, input string tag);
    logic [31:0] got;
    begin
      got = get_reg32(r);
      if (got !== exp) begin
        $display("[FAIL] %s: x%0d got=0x%08h exp=0x%08h", tag, r, got, exp);
        $fatal(1);
      end else begin
        $display("[ OK ] %s: x%0d = 0x%08h", tag, r, got);
      end
    end
  endtask

  task automatic check_mem16(input int byte_addr, input [15:0] exp, input string tag);
    int idx;
    logic [15:0] got;
    begin
      idx = (byte_addr >> 1);
      got = u_mem.mem[idx];
      if (got !== exp) begin
        $display("[FAIL] %s: mem16[0x%0h] idx=%0d got=0x%04h exp=0x%04h", tag, byte_addr, idx, got, exp);
        $fatal(1);
      end else begin
        $display("[ OK ] %s: mem16[0x%0h] = 0x%04h", tag, byte_addr, got);
      end
    end
  endtask

  task automatic check_mem32(input int byte_addr, input [31:0] exp, input string tag);
    int idx0, idx1;
    logic [15:0] lo16, hi16;
    logic [31:0] got;
    begin
      idx0 = (byte_addr >> 1);
      idx1 = ((byte_addr + 2) >> 1);
      lo16 = u_mem.mem[idx0];
      hi16 = u_mem.mem[idx1];
      got = {hi16, lo16};
      if (got !== exp) begin
        $display("[FAIL] %s: mem32[0x%0h] got=0x%08h exp=0x%08h (hi=0x%04h lo=0x%04h)", tag, byte_addr, got, exp, hi16, lo16);
        $fatal(1);
      end else begin
        $display("[ OK ] %s: mem32[0x%0h] = 0x%08h", tag, byte_addr, got);
      end
    end
  endtask

  // =============================================================================
  // Program loader
  // =============================================================================
  task automatic mem_clear_region(input int start_idx, input int words);
    int i;
    begin
      for (i = 0; i < words; i++) begin
        u_mem.mem[start_idx + i] = 16'h0000;
      end
    end
  endtask

  task automatic mem_write16(input int byte_addr, input [15:0] data16);
    begin
      u_mem.mem[byte_addr >> 1] = data16;
    end
  endtask

  task automatic mem_write32(input int byte_addr, input [31:0] data32);
    begin
      mem_write16(byte_addr,     data32[15:0]);
      mem_write16(byte_addr + 2, data32[31:16]);
    end
  endtask

  int prog_pc;

  task automatic prog_begin(input int base_addr);
    prog_pc = base_addr;
  endtask

  task automatic emit32(input [31:0] instr);
    begin
      mem_write32(prog_pc, instr);
      prog_pc += 4;
    end
  endtask

  // Load a 32-bit immediate into rd using LUI+ADDI (handles sign on low12)
  task automatic emit_li(input int rd, input [31:0] value);
    int imm12;
    int imm20;
    int sext12;
    int val_s;
    begin
      // Choose imm12 as low 12 bits, interpreted signed
      val_s  = $signed(value);
      imm12  = $signed(value[11:0]);
      sext12 = imm12; // sign-extended to 32-bit in int

      // Compute imm20 such that (imm20<<12) + sext12 == value
      imm20 = (val_s - sext12) >>> 12;

      if (($signed(value) >= -2048) && ($signed(value) <= 2047)) begin
        emit32(enc_I(7'h13, 3'h0, rd, 0, $signed(value))); // addi rd,x0,value
      end else begin
        emit32(enc_U(7'h37, rd, imm20));                   // lui
        emit32(enc_I(7'h13, 3'h0, rd, rd, imm12));         // addi
      end
    end
  endtask

  // =============================================================================
  // Run control + metrics
  // =============================================================================
  localparam int MAX_CYCLES = 200000;

  // control_fsm state encodings (must match processor.sv)
  localparam int ST_WB        = 6'd7;
  localparam int ST_WB_LOAD   = 6'd14;
  localparam int ST_WB_JUMP   = 6'd29;
  localparam int ST_SHIFT_STEP= 6'd31;

  // mem_if_adapter states
  localparam int M_IDLE = 0;
  localparam int M_ADDR = 1;
  localparam int M_DATA = 2;

  longint cycles;
  longint retired;
  longint mem_reads;
  longint mem_writes;
  longint shift_steps;

  // Basic protocol / safety assertions
  always @(posedge clk) begin
    if (rst_n) begin
      // x0 must remain 0
      if ({dut.u_core.u_rf.rf_hi[0], dut.u_core.u_rf.rf_lo[0]} !== 32'h0) begin
        $display("[FAIL] x0 not zero!");
        $fatal(1);
      end

      // uio_oe should be either all-0 or all-1 for this design
      if ((uio_oe !== 8'h00) && (uio_oe !== 8'hFF)) begin
        $display("[FAIL] uio_oe invalid: 0x%02h @%0t", uio_oe, $time);
        $fatal(1);
      end

      // mem adapter protocol (hierarchical)
      if (dut.u_core.u_memio.ms_q == M_ADDR) begin
        if (uio_oe !== 8'hFF) begin
          $display("[FAIL] M_ADDR but uio_oe != FF (0x%02h)", uio_oe);
          $fatal(1);
        end
      end

      if (dut.u_core.u_memio.ms_q == M_DATA) begin
        if (dut.u_core.u_memio.we_q) begin
          if (uio_oe !== 8'hFF) begin
            $display("[FAIL] M_DATA write but uio_oe != FF (0x%02h)", uio_oe);
            $fatal(1);
          end
        end else begin
          if (uio_oe !== 8'h00) begin
            $display("[FAIL] M_DATA read but uio_oe != 00 (0x%02h)", uio_oe);
            $fatal(1);
          end
        end
      end
    end
  end

  task automatic reset_and_init();
    begin
      rst_n = 1'b0;
      repeat (5) @(posedge clk);
      rst_n = 1'b1;
      repeat (2) @(posedge clk);

      // clear metrics
      cycles      = 0;
      retired     = 0;
      mem_reads   = 0;
      mem_writes  = 0;
      shift_steps = 0;
    end
  endtask

  // Count metrics each cycle
  always @(posedge clk) begin
    if (rst_n) begin
      cycles++;

      // memory ops: count at start of DATA phase
      if (dut.u_core.u_memio.ms_q == M_DATA) begin
        if (dut.u_core.u_memio.we_q) mem_writes++;
        else                         mem_reads++;
      end

      // shift steps: count cycles spent in S_SHIFT_STEP
      if (dut.u_core.u_ctrl.st_q == ST_SHIFT_STEP) begin
        shift_steps++;
      end

      // retired instruction approximation: count each time FSM reaches a WB-ish state
      if (dut.u_core.u_ctrl.st_q == ST_WB ||
          dut.u_core.u_ctrl.st_q == ST_WB_LOAD ||
          dut.u_core.u_ctrl.st_q == ST_WB_JUMP) begin
        retired++;
      end
    end
  end

  // Run for N cycles with timeout
  task automatic run_cycles(input int n);
    int i;
    begin
      for (i = 0; i < n; i++) begin
        @(posedge clk);
        if (cycles > MAX_CYCLES) begin
          $display("[FAIL] Timeout.");
          $fatal(1);
        end
      end
    end
  endtask

  // Run until the core reaches the "halt" instruction (jal x0,0) at DECODE
  // for `hits_required` occurrences. This is robust even though PC advances by +2 during split fetch.
  task automatic run_until_halt_decode(input int hits_required);
    int hits;
    logic [31:0] instr;
    begin
      hits = 0;
      while (hits < hits_required) begin
        @(posedge clk);
        if (cycles > MAX_CYCLES) begin
          $display("[FAIL] Timeout waiting for HALT (jal x0,0) at DECODE.");
          $fatal(1);
        end
        instr = {dut.u_core.ir_hi_q, dut.u_core.ir_lo_q};
        if (dut.u_core.u_ctrl.st_q == 6'd4 && instr === 32'h0000_006F) begin
          hits++;
        end
      end
    end
  endtask

task automatic report_metrics(input string name);
    real cpi;
    begin
      cpi = (retired != 0) ? (1.0*cycles/retired) : 0.0;
      $display("[METRIC] %s: cycles=%0d retired~=%0d memR=%0d memW=%0d shiftSteps=%0d CPI~=%0f",
               name, cycles, retired, mem_reads, mem_writes, shift_steps, cpi);
    end
  endtask

// =============================================================================
  // Directed test programs
  // =============================================================================

  // Test 1: basic add/sub + x0 + SW
  task automatic test_basic_alu_store();
    int DATA = 16'h0100;
    begin
      $display("\n=== TEST 1: basic ALU + x0 + SW ===");
      mem_clear_region(0, 2048); // clear first 4KB
      // Program at 0x0000
      prog_begin(32'h0000);

      emit32(enc_I(7'h13,3'h0, 1,0,  5));     // addi x1,x0,5
      emit32(enc_I(7'h13,3'h0, 2,0, -3));     // addi x2,x0,-3
      emit32(enc_R(7'h33,3'h0, 7'h00, 3,1,2));// add  x3,x1,x2 => 2
      emit32(enc_R(7'h33,3'h0, 7'h20, 4,1,2));// sub  x4,x1,x2 => 8
      emit32(enc_I(7'h13,3'h0, 0,0, 123));    // addi x0,x0,123 (should have no effect)

      emit32(enc_I(7'h13,3'h0, 5,0, DATA));   // addi x5,x0,0x100
      emit32(enc_S(7'h23,3'h2, 5,4, 0));      // sw x4,0(x5)

      emit32(enc_J(7'h6F, 0, 0));             // jal x0,0 (tight loop)

      reset_and_init();
      run_until_halt_decode(3);

      check_reg32(1, 32'd5, "basic");
      check_reg32(2, 32'hFFFF_FFFD, "basic");
      check_reg32(3, 32'd2, "basic");
      check_reg32(4, 32'd8, "basic");
      check_reg32(0, 32'd0, "x0");
      check_mem32(DATA, 32'd8, "sw");
      report_metrics("TEST1");
    end
  endtask

  // Test 2: sign/zero extension loads (LB/LBU/LH/LHU/LW)
  task automatic test_load_ext();
    int BASE = 16'h0200;
    begin
      $display("\n=== TEST 2: LB/LBU/LH/LHU/LW sign/zero extension ===");
      mem_clear_region(0, 4096);

      // Seed data:
      // At 0x0200 halfword = 0x8001 (bytes: 0x01,0x80)
      // At 0x0202 halfword = 0x7FEE (bytes: 0xEE,0x7F)
      mem_write16(BASE,     16'h8001);
      mem_write16(BASE + 2, 16'h7FEE);

      prog_begin(32'h0000);
      emit32(enc_I(7'h13,3'h0, 5,0, BASE));   // addi x5,x0,0x200

      emit32(enc_I(7'h03,3'h0, 1,5, 0));      // lb  x1,0(x5)  -> 0x00000001 (since low byte=0x01)
      emit32(enc_I(7'h03,3'h4, 2,5, 1));      // lbu x2,1(x5)  -> 0x00000080
      emit32(enc_I(7'h03,3'h1, 3,5, 0));      // lh  x3,0(x5)  -> 0xFFFF8001
      emit32(enc_I(7'h03,3'h5, 4,5, 0));      // lhu x4,0(x5)  -> 0x00008001
      emit32(enc_I(7'h03,3'h2, 6,5, 0));      // lw  x6,0(x5)  -> 0x7FEE8001

      emit32(enc_J(7'h6F, 0, 0));             // loop

      reset_and_init();
      run_until_halt_decode(3);

      check_reg32(1, 32'h0000_0001, "load_ext");
      check_reg32(2, 32'h0000_0080, "load_ext");
      check_reg32(3, 32'hFFFF_8001, "load_ext");
      check_reg32(4, 32'h0000_8001, "load_ext");
      check_reg32(6, 32'h7FEE_8001, "load_ext");
      report_metrics("TEST2");
    end
  endtask

  // Test 3: SB/SH/SW (SB requires RMW on this bus)
    // Test 3: SB/SH/SW (SB requires RMW on this bus)
  task automatic test_store_sizes();
    int BASE = 16'h0300;
    begin
      $display("\n=== TEST 3: SB/SH/SW (RMW byte store) ===");
      mem_clear_region(0, 4096);

      // Seed: halfword at 0x0300 = 0xA1B2
      mem_write16(BASE,     16'hA1B2);
      mem_write16(BASE + 2, 16'h0000);

      prog_begin(32'h0000);
      emit32(enc_I(7'h13,3'h0, 5,0, BASE));   // addi x5,x0,0x300

      emit32(enc_I(7'h13,3'h0, 1,0, 8'hCC));  // addi x1,x0,0xCC
      emit32(enc_S(7'h23,3'h0, 5,1, 0));      // sb x1,0(x5) -> low byte = CC (A1CC)

      emit32(enc_I(7'h13,3'h0, 1,0, 8'hDD));  // addi x1,x0,0xDD
      emit32(enc_S(7'h23,3'h0, 5,1, 1));      // sb x1,1(x5) -> high byte = DD (DDCC)

      emit32(enc_I(7'h03,3'h1, 2,5, 0));      // lh x2,0(x5) -> 0xFFFFDDCC

      emit_li(1, 32'h0000_1122);
      emit32(enc_S(7'h23,3'h1, 5,1, 0));      // sh x1,0(x5) -> 0x1122

      emit32(enc_I(7'h03,3'h1, 3,5, 0));      // lh x3,0(x5) -> 0x00001122

      emit_li(1, 32'h0000_3344);
      emit32(enc_S(7'h23,3'h2, 5,1, 0));      // sw x1,0(x5) -> [0x0300]=0x3344, [0x0302]=0x0000

      emit32(enc_I(7'h03,3'h2, 4,5, 0));      // lw x4,0(x5) -> 0x00003344

      emit32(enc_J(7'h6F, 0, 0));             // loop

      reset_and_init();
      run_until_halt_decode(3);

      check_reg32(2, 32'hFFFF_DDCC, "store_sizes (after SB/SB)");
      check_reg32(3, 32'h0000_1122, "store_sizes (after SH)");
      check_reg32(4, 32'h0000_3344, "store_sizes (after SW/LW)");

      check_mem16(BASE,     16'h3344, "store_sizes mem");
      check_mem16(BASE + 2, 16'h0000, "store_sizes mem");
      report_metrics("TEST3");
    end
  endtask

  // Test 4: branches signed/unsigned correctness
    // Test 4: branches signed/unsigned correctness
  task automatic test_branches();
    begin
      $display("\n=== TEST 4: branches signed/unsigned (BLT/BLTU/BGEU) ===");
      mem_clear_region(0, 4096);

      prog_begin(32'h0000);

      emit32(enc_I(7'h13,3'h0, 1,0, -1));     // x1 = -1
      emit32(enc_I(7'h13,3'h0, 2,0,  1));     // x2 = 1
      emit32(enc_I(7'h13,3'h0, 3,0,  0));     // x3 = 0 (error accumulator)

      // 0x000C: blt x1,x2, L1 (should TAKE)  L1=0x0014 offset=+8
      emit32(enc_B(7'h63,3'h4, 1,2, 13'sd8)); // blt
      // 0x0010: error if not taken
      emit32(enc_I(7'h13,3'h0, 3,3,  1));     // x3 += 1

      // 0x0014 L1:
      // bltu x1,x2,BAD1 (should NOT take) BAD1=0x001C offset=+8
      emit32(enc_B(7'h63,3'h6, 1,2, 13'sd8)); // bltu
      // 0x0018: jump over BAD1 to CONT (CONT=0x0020 offset=+8)
      emit32(enc_J(7'h6F, 0, 21'sd8));        // jal x0, CONT

      // 0x001C BAD1: only if BLTU wrongly taken
      emit32(enc_I(7'h13,3'h0, 3,3,  2));     // x3 += 2

      // 0x0020 CONT:
      // bgeu x1,x2,L2 (should TAKE) L2=0x0028 offset=+8
      emit32(enc_B(7'h63,3'h7, 1,2, 13'sd8)); // bgeu
      // 0x0024 error if not taken
      emit32(enc_I(7'h13,3'h0, 3,3,  4));     // x3 += 4
      // 0x0028 L2:
      emit32(enc_J(7'h6F, 0, 21'sd0));        // loop

      reset_and_init();
      run_until_halt_decode(3);

      check_reg32(3, 32'h0000_0000, "branches (x3 must be 0)");
      report_metrics("TEST4");
    end
  endtask

  // Test 5: serial shifts (SRA/SRL/SLL + shamt masking)
  task automatic test_shifts();
    begin
      $display("\n=== TEST 5: serial shifts (SRA/SRL/SLL + shamt masking) ===");
      mem_clear_region(0, 4096);

      prog_begin(32'h0000);

      // x1 = 0x80000001
      emit_li(1, 32'h8000_0001);

      // srai x2,x1,1 => 0xC0000000
      emit32(enc_I_sh(7'h13, 3'h5, 7'h20, 2, 1, 1)); // srai

      // srli x3,x1,1 => 0x40000000
      emit32(enc_I_sh(7'h13, 3'h5, 7'h00, 3, 1, 1)); // srli

      // slli x4,x3,2 => 0x00000000? Wait 0x40000000<<2 = 0x00000000 (overflow) -> 0x00000000
      emit32(enc_I_sh(7'h13, 3'h1, 7'h00, 4, 3, 2)); // slli

      // shamt masking: x5=40 (0b101000) => shift by 8
      emit32(enc_I(7'h13,3'h0, 5,0, 40));            // addi x5,x0,40
      // srl x6,x1,x5 (R-type SRL)
      emit32(enc_R(7'h33,3'h5, 7'h00, 6, 1, 5));     // srl x6,x1,x5

      emit32(enc_J(7'h6F, 0, 0));                     // loop

      reset_and_init();
      run_until_halt_decode(3); // shifts are multi-cycle

      check_reg32(1, 32'h8000_0001, "shifts");
      check_reg32(2, 32'hC000_0000, "shifts");
      check_reg32(3, 32'h4000_0000, "shifts");
      check_reg32(4, 32'h0000_0000, "shifts");
      check_reg32(6, 32'h0080_0000, "shifts (mask shamt)");
      $display("[METRIC] shift_steps=%0d (should be >0 due to serial shifting)", shift_steps);
      if (shift_steps == 0) begin
        $display("[FAIL] shift_steps=0; serial shifter may not be active.");
        $fatal(1);
      end
      report_metrics("TEST5");
    end
  endtask

  // Optional: randomized ALU smoke (no branches needed)
  task automatic test_random_alu_smoke();
    int i;
    int seed;
    int DATA = 16'h0400;
    logic [31:0] a, b, exp;    int base;

    begin
      $display("\n=== TEST 6: randomized ALU smoke (ADD/SUB/XOR/AND/OR/SLT/SLTU) ===");
      mem_clear_region(0, 8192);
      seed = 32'hC0FFEE01;

      prog_begin(32'h0000);
      emit32(enc_I(7'h13,3'h0, 5,0, DATA)); // x5 = base pointer 0x400

      // We will do 10 cases; store results to memory so TB can check.
      for (i = 0; i < 10; i++) begin
        seed = seed * 32'h0019660D + 32'h3C6EF35F;
        a = seed ^ (seed << 13);
        seed = seed * 32'h0019660D + 32'h3C6EF35F;
        b = seed ^ (seed >> 7);

        // Load operands into x1 and x2
        emit_li(1, a);
        emit_li(2, b);

        // ADD -> x3
        emit32(enc_R(7'h33,3'h0,7'h00, 3,1,2));
        // SUB -> x4
        emit32(enc_R(7'h33,3'h0,7'h20, 4,1,2));
        // XOR -> x6
        emit32(enc_R(7'h33,3'h4,7'h00, 6,1,2));
        // AND -> x7
        emit32(enc_R(7'h33,3'h7,7'h00, 7,1,2));
        // OR  -> x8
        emit32(enc_R(7'h33,3'h6,7'h00, 8,1,2));
        // SLT -> x9
        emit32(enc_R(7'h33,3'h2,7'h00, 9,1,2));
        // SLTU -> x10
        emit32(enc_R(7'h33,3'h3,7'h00, 10,1,2));

        // Store a small signature of results (low 16 bits) sequentially
        // sw x3, 0(x5) ; sw x4,4(x5) ; sw x6,8(x5) ...
        emit32(enc_S(7'h23,3'h2, 5,3,  0));
        emit32(enc_S(7'h23,3'h2, 5,4,  4));
        emit32(enc_S(7'h23,3'h2, 5,6,  8));
        emit32(enc_S(7'h23,3'h2, 5,7, 12));
        emit32(enc_S(7'h23,3'h2, 5,8, 16));
        emit32(enc_S(7'h23,3'h2, 5,9, 20));
        emit32(enc_S(7'h23,3'h2, 5,10,24));

        // advance pointer by 28 bytes
        emit32(enc_I(7'h13,3'h0, 5,5, 28));
      end

      emit32(enc_J(7'h6F, 0, 0)); // loop

      reset_and_init();
      run_until_halt_decode(5);

      // Recompute expected and check memory
      seed = 32'hC0FFEE01;
      for (i = 0; i < 10; i++) begin
        seed = seed * 32'h0019660D + 32'h3C6EF35F;
        a = seed ^ (seed << 13);
        seed = seed * 32'h0019660D + 32'h3C6EF35F;
        b = seed ^ (seed >> 7);

        // Base address for this case
        base = DATA + (i * 28);

        check_mem32(base +  0, a + b, "rand/add");
        check_mem32(base +  4, a - b, "rand/sub");
        check_mem32(base +  8, a ^ b, "rand/xor");
        check_mem32(base + 12, a & b, "rand/and");
        check_mem32(base + 16, a | b, "rand/or");
        exp = ($signed(a) < $signed(b)) ? 32'd1 : 32'd0;
        check_mem32(base + 20, exp, "rand/slt");
        exp = (a < b) ? 32'd1 : 32'd0;
        check_mem32(base + 24, exp, "rand/sltu");
      end
      report_metrics("TEST6");
    end
  endtask

  // =============================================================================
  // Main
  // =============================================================================
  initial begin
    $dumpfile("waves.vcd");
    $dumpvars(0, tb_microcore16);

    // Run suite
    test_basic_alu_store();
    test_load_ext();
    test_store_sizes();
    test_branches();
    test_shifts();
    test_random_alu_smoke();

    $display("\n=== ALL TESTS PASSED ===");
    $display("[METRIC][LAST] cycles=%0d retired~=%0d mem_reads=%0d mem_writes=%0d shift_steps=%0d CPI~=%0f",
             cycles, retired, mem_reads, mem_writes, shift_steps,
             (retired != 0) ? (1.0*cycles/retired) : 0.0);

    $finish;
  end

endmodule
