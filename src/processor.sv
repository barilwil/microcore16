// ================================================================
// MicroCore-16 (A1 + B2) — Skeleton Verilog/SystemVerilog Module Set
// - 16-bit split datapath RV32E-style core (16 regs x 32-bit, x0=0)
// - 64KB memory space (16-bit byte address), 16-bit memory port
// - valid/ready handshake (B2)
// ================================================================
// NOTE: This is a skeleton: ports, registers, state machine structure,
//       and “where logic goes”. You’ll fill in decode + muxing + ALU ops.
// ================================================================



// ================================================================
// Global types (visible to all modules in this file)
// ================================================================
typedef enum logic [3:0] {
  ALU_ADD  = 4'd0,
  ALU_SUB  = 4'd1,
  ALU_AND  = 4'd2,
  ALU_OR   = 4'd3,
  ALU_XOR  = 4'd4,
  ALU_PASS = 4'd5
} aluop_t;

// ================================================================
// 1) Top-level wrapper (Tiny Tapeout style-ish)
// ================================================================
module microcore16_top (
  input  logic        clk,
  input  logic        rst_n,

  // Tiny Tapeout-like I/O (adjust to your exact TT template)
  input  logic [7:0]  ui_in,      // dedicated inputs
  output logic [7:0]  uo_out,     // dedicated outputs
  input  logic [7:0]  uio_in,     // bidir inputs
  output logic [7:0]  uio_out,    // bidir outputs
  output logic [7:0]  uio_oe      // bidir output enables (1=drive)
);

  // -------------------------
  // Core architectural regs
  // -------------------------
  logic [31:0] pc_q, pc_d;

  logic [15:0] ir_lo_q, ir_lo_d;
  logic [15:0] ir_hi_q, ir_hi_d;
  logic        ir_lo_we, ir_hi_we;

  // Operand & result latches (split halves)
  logic [15:0] a_lo_q, a_hi_q, b_lo_q, b_hi_q;
  logic        ab_latch_we; // latch A/B from regfile during DECODE (or as needed)

  logic [15:0] r_lo_q, r_hi_q;
  logic        r_lo_we, r_hi_we;
  logic        r_use_ext;
  logic [15:0] r_ext_lo, r_ext_hi;

  logic        carry_q, carry_d;
  logic        carry_we;

  // Memory buffers for load
  logic [15:0] mem_buf_lo_q, mem_buf_hi_q;
  logic        mem_buf_lo_we, mem_buf_hi_we;

  // Address calc latches (optional but helpful)
  logic [15:0] addr_lo_q, addr_hi_q;
  logic        addr_lo_we, addr_hi_we;

  // -------------------------
  // Instruction fields
  // -------------------------
  logic [31:0] ir32;
  assign ir32 = {ir_hi_q, ir_lo_q};

  logic [6:0]  opcode;
  logic [2:0]  funct3;
  logic [6:0]  funct7;

  // RISC-V encodes registers as 5-bit fields; RV32E restricts them to 0..15.
  logic [4:0]  rs1_5, rs2_5, rd_5;
  logic [3:0]  rs1, rs2, rd;
  logic        illegal_rs1, illegal_rs2, illegal_rd;

  // Shift-immediate amount (SLLI/SRLI/SRAI)
  logic [4:0]  shamt5;

  assign opcode = ir32[6:0];
  assign funct3 = ir32[14:12];
  assign funct7 = ir32[31:25];

  assign rd_5  = ir32[11:7];
  assign rs1_5 = ir32[19:15];
  assign rs2_5 = ir32[24:20];

  assign rd    = rd_5[3:0];
  assign rs1   = rs1_5[3:0];
  assign rs2   = rs2_5[3:0];

  assign illegal_rd  = rd_5[4];
  assign illegal_rs1 = rs1_5[4];
  assign illegal_rs2 = rs2_5[4];

  assign shamt5 = ir32[24:20];

// -------------------------
  // Immediate generation
  // -------------------------
  logic [31:0] imm32;
  logic [15:0] imm_lo, imm_hi;
  assign imm_lo = imm32[15:0];
  assign imm_hi = imm32[31:16];

  // TODO: fill in proper immediate decoding (I/S/B/U/J types)
  // RV32I/E
  always_comb begin
    unique case (opcode)
        7'b001_0011, // OP-IMM (I-type)
        7'b000_0011, // LOAD (I-type)    
        7'b110_0111: // JALR (I-type)
            imm32 = {{20{ir32[31]}}, ir32[31:20]}; // I-type
        
        7'b010_0011: // STORE (S-type)
            imm32 = {{20{ir32[31]}}, ir32[31:25], ir32[11:7]}; // S-type

        7'b110_0011: // BRANCH (B-type)
            imm32 = {{19{ir32[31]}}, ir32[31], ir32[7], ir32[30:25], ir32[11:8], 1'b0}; // B-type
        
        7'b011_0111, // LUI (U-type)
        7'b001_0111: // AUIPC (U-type)
            imm32 = {ir32[31:12], 12'b0}; // U-type
        
        7'b110_1111: // JAL (J-type)
            imm32 = {{11{ir32[31]}}, ir32[31], ir32[19:12], ir32[20], ir32[30:21], 1'b0}; // J-type
        
        default: imm32 = 32'h0; // default
    endcase
  end

  // -------------------------
  // Regfile wiring
  // -------------------------
  logic [15:0] rf_rs1_lo, rf_rs1_hi, rf_rs2_lo, rf_rs2_hi;
  logic        rf_we;
  logic [3:0]  rf_waddr;
  logic [15:0] rf_wdata_lo, rf_wdata_hi;

  regfile_rv32e_split u_rf (
    .clk       (clk),
    .rst_n     (rst_n),

    .raddr1    (rs1),
    .rdata1_lo (rf_rs1_lo),
    .rdata1_hi (rf_rs1_hi),

    .raddr2    (rs2),
    .rdata2_lo (rf_rs2_lo),
    .rdata2_hi (rf_rs2_hi),

    .we        (rf_we),
    .waddr     (rf_waddr),
    .wdata_lo  (rf_wdata_lo),
    .wdata_hi  (rf_wdata_hi)
  );

  // Latch operands (typical: in DECODE)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_lo_q <= 16'h0; a_hi_q <= 16'h0;
      b_lo_q <= 16'h0; b_hi_q <= 16'h0;
    end else if (ab_latch_we) begin
      a_lo_q <= rf_rs1_lo; a_hi_q <= rf_rs1_hi;
      b_lo_q <= rf_rs2_lo; b_hi_q <= rf_rs2_hi;
    end
  end

  // -------------------------
  // ALU16 wiring + input muxes
  // -------------------------
// Compute the PC of the current instruction (pc_q points to next instruction at DECODE)
  logic [31:0] pc_inst;
  assign pc_inst = pc_q - 32'd4;

  aluop_t   alu_op;
  logic [15:0] alu_a, alu_b, alu_y;
  logic        alu_cin, alu_cout;
  logic        alu_z, alu_n;

  alu16 u_alu16 (
    .a    (alu_a),
    .b    (alu_b),
    .op   (alu_op),
    .cin  (alu_cin),
    .y    (alu_y),
    .cout (alu_cout),
    .z    (alu_z),
    .n    (alu_n)
  );

  // These mux selects are driven by the FSM
  logic [2:0] alu_a_sel, alu_b_sel;
  logic       alu_use_carry_in; // for hi-half add/sub

  // Example muxing (expand as needed)
  always_comb begin
    // default
    alu_a = 16'h0;
    alu_b = 16'h0;
    alu_cin = (alu_use_carry_in) ? carry_q : 1'b0;

    unique case (alu_a_sel)
      3'd0: alu_a = a_lo_q;          // rs1 low
      3'd1: alu_a = a_hi_q;          // rs1 high
      3'd2: alu_a = pc_q[15:0];      // PC (current)
      3'd3: alu_a = pc_q[31:16];
      3'd4: alu_a = pc_inst[15:0];   // PC of current instruction (pc_q - 4)
      3'd5: alu_a = pc_inst[31:16];
      3'd6: alu_a = addr_lo_q;
      3'd7: alu_a = addr_hi_q;
      default: alu_a = 16'h0;
    endcase


    unique case (alu_b_sel)
      3'd0: alu_b = b_lo_q;
      3'd1: alu_b = b_hi_q;
      3'd2: alu_b = 16'd2;         // PC + 2
      3'd3: alu_b = imm_lo;
      3'd4: alu_b = imm_hi;
      default: alu_b = 16'h0;
    endcase
  end

  // Result latches
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_lo_q <= 16'h0; r_hi_q <= 16'h0;
      carry_q <= 1'b0;
    end else begin
      if (r_lo_we) r_lo_q <= (r_use_ext ? r_ext_lo : alu_y);
      if (r_hi_we) r_hi_q <= (r_use_ext ? r_ext_hi : alu_y);
      if (carry_we) carry_q <= alu_cout;
    end
  end

  // -------------------------
  // PC + IR regs
  // -------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc_q    <= 32'h0;
      ir_lo_q <= 16'h0;
      ir_hi_q <= 16'h0;
    end else begin
      pc_q <= pc_d;
      if (ir_lo_we) ir_lo_q <= ir_lo_d;
      if (ir_hi_we) ir_hi_q <= ir_hi_d;
    end
  end

  // Address calc latches (optional)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      addr_lo_q <= 16'h0;
      addr_hi_q <= 16'h0;
    end else begin
      if (addr_lo_we) addr_lo_q <= alu_y;
      if (addr_hi_we) addr_hi_q <= alu_y;
    end
  end


  // -------------------------
  // Memory interface (core-side signals)
  // -------------------------
  // -------------------------
  logic        mem_valid, mem_ready, mem_we;
  logic [15:0] mem_addr16, mem_wdata16, mem_rdata16;

  // Load buffers
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mem_buf_lo_q <= 16'h0;
      mem_buf_hi_q <= 16'h0;
    end else begin
      if (mem_buf_lo_we) mem_buf_lo_q <= mem_rdata16;
      if (mem_buf_hi_we) mem_buf_hi_q <= mem_rdata16;
    end
  end

  // -------------------------

  // Adapter to TT pins
  mem_if_adapter_tt u_memio (
    .clk        (clk),
    .rst_n      (rst_n),

    .mem_valid  (mem_valid),
    .mem_ready  (mem_ready),
    .mem_we     (mem_we),
    .mem_addr16 (mem_addr16),
    .mem_wdata16(mem_wdata16),
    .mem_rdata16(mem_rdata16),

    .ui_in      (ui_in),
    .uo_out     (uo_out),
    .uio_in     (uio_in),
    .uio_out    (uio_out),
    .uio_oe     (uio_oe)
  );

  // -------------------------
  // Control FSM → drives everything
  // -------------------------
  control_fsm u_ctrl (
    .clk        (clk),
    .rst_n      (rst_n),

    // Instr fields / decode inputs
    .opcode     (opcode),
    .funct3     (funct3),
    .funct7     (funct7),
    .rs1        (rs1),
    .rs2        (rs2),
    .rd         (rd),

    .illegal_rs1(illegal_rs1),
    .illegal_rs2(illegal_rs2),
    .illegal_rd (illegal_rd),

    .shamt5     (shamt5),

    // ALU status
    .alu_z      (alu_z),
    .alu_n      (alu_n),
    .alu_cout   (alu_cout),
    .alu_y      (alu_y),

    // handshake
    .mem_ready  (mem_ready),

    // immediates + memory data
    .imm_lo     (imm_lo),
    .imm_hi     (imm_hi),
    .mem_rdata16(mem_rdata16),

    // --- outputs: architectural control ---
    .pc_q       (pc_q),
    .pc_d       (pc_d),

    .ir_lo_we   (ir_lo_we),
    .ir_hi_we   (ir_hi_we),
    .ir_lo_d    (ir_lo_d),
    .ir_hi_d    (ir_hi_d),

    .ab_latch_we(ab_latch_we),

    .alu_op     (alu_op),
    .alu_a_sel  (alu_a_sel),
    .alu_b_sel  (alu_b_sel),
    .alu_use_carry_in(alu_use_carry_in),

    .r_lo_we    (r_lo_we),
    .r_hi_we    (r_hi_we),
    .carry_we   (carry_we),

    .r_use_ext  (r_use_ext),
    .r_ext_lo   (r_ext_lo),
    .r_ext_hi   (r_ext_hi),

    .addr_lo_we (addr_lo_we),
    .addr_hi_we (addr_hi_we),

    .mem_buf_lo_we(mem_buf_lo_we),
    .mem_buf_hi_we(mem_buf_hi_we),

    .rf_we      (rf_we),
    .rf_waddr   (rf_waddr),
    .rf_wdata_lo(rf_wdata_lo),
    .rf_wdata_hi(rf_wdata_hi),

    // memory request outputs
    .mem_valid  (mem_valid),
    .mem_we     (mem_we),
    .mem_addr16 (mem_addr16),
    .mem_wdata16(mem_wdata16),

    // datapath inputs the controller may need
    .a_lo       (a_lo_q),
    .a_hi       (a_hi_q),
    .b_lo       (b_lo_q),
    .b_hi       (b_hi_q),
    .r_lo       (r_lo_q),
    .r_hi       (r_hi_q),
    .addr_lo_q  (addr_lo_q),
    .addr_hi_q  (addr_hi_q),
    .mem_buf_lo (mem_buf_lo_q),
    .mem_buf_hi (mem_buf_hi_q)
  );
endmodule

// ================================================================
// 2) Control FSM (A1 + B2 state sequencing) — COMPLETED STARTER VERSION
// ================================================================
// Supports (starter set):
//  - R-type: ADD/SUB, AND/OR/XOR
//  - I-type: ADDI, ANDI/ORI/XORI
//  - LW/SW  : word (funct3=010) using 2x16-bit transfers
//  - BEQ/BNE: basic compare (equality) and PC-relative branch
//
// Assumptions / conventions:
//  - PC has already advanced by +4 after FETCH_HI_RESP, so "pc_q" at DECODE
//    is the address of the *next* instruction. Branch target uses that base.
//  - Memory is 64KB (A1): we only use addr_lo_q for mem_addr16.
//  - Handshake (B2): mem_valid held for one cycle in *_REQ; wait mem_ready.
//  - This FSM directly writes IR halves from mem_rdata16 (ir_*_d = mem_rdata16).
//
// NOTE: Branch compare uses SUB. Depending on your SUB carry/borrow convention,
// you may later refine alu_use_carry_in for SUB. Equality check uses alu_y==0.
// ================================================================
module control_fsm (
  input  logic        clk,
  input  logic        rst_n,

  input  logic [6:0]  opcode,
  input  logic [2:0]  funct3,
  input  logic [6:0]  funct7,
  input  logic [3:0]  rs1, rs2, rd,

  // RV32E legality checks (bit4 of the encoded 5-bit reg index)
  input  logic        illegal_rs1,
  input  logic        illegal_rs2,
  input  logic        illegal_rd,

  // shift immediate amount (ir[24:20])
  input  logic [4:0]  shamt5,

  // ALU status
  input  logic        alu_z,
  input  logic        alu_n,
  input  logic        alu_cout,     // carry-out for ADD, borrow-out for SUB
  input  logic [15:0] alu_y,        // current 16-bit ALU result (for PC redirect timing)

  // memory handshake
  input  logic        mem_ready,

  // immediate (already expanded to full 32b in top, then split)
  input  logic [15:0] imm_lo,
  input  logic [15:0] imm_hi,

  // memory read data (held by mem adapter after a completed read)
  input  logic [15:0] mem_rdata16,

  // PC / IR
  input  logic [31:0] pc_q,         // note: at DECODE this is already PC+4
  output logic [31:0] pc_d,

  output logic        ir_lo_we,
  output logic        ir_hi_we,
  output logic [15:0] ir_lo_d,
  output logic [15:0] ir_hi_d,

  // latch controls (latch A/B from regfile in top)
  output logic        ab_latch_we,

  // ALU controls (drives top-level ALU muxes)
  output aluop_t      alu_op,
  output logic [2:0]  alu_a_sel,
  output logic [2:0]  alu_b_sel,
  output logic        alu_use_carry_in,

  output logic        r_lo_we,
  output logic        r_hi_we,
  output logic        carry_we,

  // direct write to r regs (used for serial shifter / 1-cycle results like SLT)
  output logic        r_use_ext,
  output logic [15:0] r_ext_lo,
  output logic [15:0] r_ext_hi,

  output logic        addr_lo_we,
  output logic        addr_hi_we,

  output logic        mem_buf_lo_we,
  output logic        mem_buf_hi_we,

  // regfile writeback
  output logic        rf_we,
  output logic [3:0]  rf_waddr,
  output logic [15:0] rf_wdata_lo,
  output logic [15:0] rf_wdata_hi,

  // memory request
  output logic        mem_valid,
  output logic        mem_we,
  output logic [15:0] mem_addr16,
  output logic [15:0] mem_wdata16,

  // datapath values
  input  logic [15:0] a_lo, a_hi, b_lo, b_hi,
  input  logic [15:0] r_lo, r_hi,
  input  logic [15:0] addr_lo_q,
  input  logic [15:0] addr_hi_q,
  input  logic [15:0] mem_buf_lo, mem_buf_hi
);

  // ------------------------------------------------------------
  // State machine
  // ------------------------------------------------------------
  typedef enum logic [5:0] {
    S_FETCH_LO_REQ   = 6'd0,
    S_FETCH_LO_RESP  = 6'd1,
    S_FETCH_HI_REQ   = 6'd2,
    S_FETCH_HI_RESP  = 6'd3,
    S_DECODE         = 6'd4,

    S_EXEC_LO        = 6'd5,
    S_EXEC_HI        = 6'd6,
    S_WB             = 6'd7,

    S_ADDR_LO        = 6'd8,
    S_ADDR_HI        = 6'd9,

    S_MEM_RD_LO_REQ  = 6'd10,
    S_MEM_RD_LO_RESP = 6'd11,
    S_MEM_RD_HI_REQ  = 6'd12,
    S_MEM_RD_HI_RESP = 6'd13,
    S_WB_LOAD        = 6'd14,

    S_MEM_WR_LO_REQ  = 6'd15,
    S_MEM_WR_LO_RESP = 6'd16,
    S_MEM_WR_HI_REQ  = 6'd17,
    S_MEM_WR_HI_RESP = 6'd18,

    S_SB_RD_REQ      = 6'd19,
    S_SB_RD_RESP     = 6'd20,
    S_SB_WR_REQ      = 6'd21,
    S_SB_WR_RESP     = 6'd22,

    S_CMP_LO         = 6'd23,
    S_CMP_HI         = 6'd24,

    S_BR_TGT_LO      = 6'd25,
    S_BR_TGT_HI      = 6'd26,

    S_JAL_TGT_LO     = 6'd27,
    S_JAL_TGT_HI     = 6'd28,
    S_WB_JUMP        = 6'd29,

    S_SHIFT_INIT     = 6'd30,
    S_SHIFT_STEP     = 6'd31
  } state_t;

  state_t st_q, st_d;

  // ------------------------------------------------------------
  // Latched "instruction intent"
  // ------------------------------------------------------------
  typedef enum logic [3:0] {
    K_NOP    = 4'd0,
    K_ALU    = 4'd1,  // ADD/SUB/AND/OR/XOR (reg or imm)
    K_LOAD   = 4'd2,
    K_STORE  = 4'd3,
    K_BRANCH = 4'd4,
    K_JAL    = 4'd5,
    K_JALR   = 4'd6,
    K_LUI    = 4'd7,
    K_AUIPC  = 4'd8,
    K_SHIFT  = 4'd9,
    K_SLT    = 4'd10
  } kind_t;

  kind_t kind_q, kind_d;

  aluop_t op_q, op_d;
  logic   use_imm_q, use_imm_d;      // B comes from imm for ALU/SLT
  logic   wb_en_q, wb_en_d;

  // for SLT / BRANCH comparisons
  logic   slt_unsigned_q, slt_unsigned_d;
  logic [2:0] br_funct3_q, br_funct3_d;
  logic   eq_lo_q, eq_lo_d;

  // mem sizes (funct3)
  logic [2:0] mem_funct3_q, mem_funct3_d;

  // shift controls
  logic       sh_dir_q, sh_dir_d;        // 0=left, 1=right
  logic       sh_arith_q, sh_arith_d;    // only for right shifts: 1=SRA, 0=SRL
  logic       sh_amt_is_imm_q, sh_amt_is_imm_d;
  logic [4:0] sh_count_q, sh_count_d;

  // ------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------
  logic is_op, is_opimm, is_load, is_store, is_branch, is_jal, is_jalr, is_lui, is_auipc;
  always_comb begin
    is_op     = (opcode == 7'b0110011);
    is_opimm  = (opcode == 7'b0010011);
    is_load   = (opcode == 7'b0000011);
    is_store  = (opcode == 7'b0100011);
    is_branch = (opcode == 7'b1100011);
    is_jal    = (opcode == 7'b1101111);
    is_jalr   = (opcode == 7'b1100111);
    is_lui    = (opcode == 7'b0110111);
    is_auipc  = (opcode == 7'b0010111);
  end

  // PC of current instruction (since pc_q is already PC+4 at DECODE)
  logic [31:0] pc_inst;
  assign pc_inst = pc_q - 32'd4;

  // ------------------------------------------------------------
  // Sequential state / latches
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st_q            <= S_FETCH_LO_REQ;

      kind_q          <= K_NOP;
      op_q            <= ALU_ADD;
      use_imm_q       <= 1'b0;
      wb_en_q         <= 1'b0;

      slt_unsigned_q  <= 1'b0;
      br_funct3_q     <= 3'b000;
      eq_lo_q         <= 1'b0;

      mem_funct3_q    <= 3'b010;

      sh_dir_q        <= 1'b0;
      sh_arith_q      <= 1'b0;
      sh_amt_is_imm_q <= 1'b0;
      sh_count_q      <= 5'd0;
    end else begin
      st_q            <= st_d;

      kind_q          <= kind_d;
      op_q            <= op_d;
      use_imm_q       <= use_imm_d;
      wb_en_q         <= wb_en_d;

      slt_unsigned_q  <= slt_unsigned_d;
      br_funct3_q     <= br_funct3_d;
      eq_lo_q         <= eq_lo_d;

      mem_funct3_q    <= mem_funct3_d;

      sh_dir_q        <= sh_dir_d;
      sh_arith_q      <= sh_arith_d;
      sh_amt_is_imm_q <= sh_amt_is_imm_d;
      sh_count_q      <= sh_count_d;
    end
  end

  // ------------------------------------------------------------
  // Combinational FSM
  // ------------------------------------------------------------
  // shared temporaries (tool-friendly: declared outside case items)
  logic        cmp_eq_full, cmp_lt_u, cmp_lt_s, cmp_take;
  logic [31:0] load32;
  logic [7:0]  load_b8;
  logic [15:0] load_h16;
  logic [4:0]  shamt_sel;
  logic [15:0] sh_nlo, sh_nhi;

  always_comb begin
    // defaults
    st_d = st_q;

    pc_d = pc_q;

    // IR write controls
    ir_lo_we = 1'b0;
    ir_hi_we = 1'b0;
    ir_lo_d  = mem_rdata16;
    ir_hi_d  = mem_rdata16;

    // latch A/B in DECODE
    ab_latch_we = 1'b0;

    // ALU control defaults
    alu_op           = ALU_ADD;
    alu_a_sel        = 3'd0;
    alu_b_sel        = 3'd0;
    alu_use_carry_in = 1'b0;

    // datapath latch enables
    r_lo_we   = 1'b0;
    r_hi_we   = 1'b0;
    carry_we  = 1'b0;

    r_use_ext = 1'b0;
    r_ext_lo  = 16'h0;
    r_ext_hi  = 16'h0;

    addr_lo_we = 1'b0;
    addr_hi_we = 1'b0;

    mem_buf_lo_we = 1'b0;
    mem_buf_hi_we = 1'b0;

    // regfile writeback
    rf_we       = 1'b0;
    rf_waddr    = 4'd0;
    rf_wdata_lo = 16'h0;
    rf_wdata_hi = 16'h0;

    // memory request
    mem_valid   = 1'b0;
    mem_we      = 1'b0;
    mem_addr16  = 16'h0;
    mem_wdata16 = 16'h0;

    // latch defaults
    kind_d          = kind_q;
    op_d            = op_q;
    use_imm_d       = use_imm_q;
    wb_en_d         = wb_en_q;

    slt_unsigned_d  = slt_unsigned_q;
    br_funct3_d     = br_funct3_q;
    eq_lo_d         = eq_lo_q;

    mem_funct3_d    = mem_funct3_q;

    sh_dir_d        = sh_dir_q;
    sh_arith_d      = sh_arith_q;
    sh_amt_is_imm_d = sh_amt_is_imm_q;
    sh_count_d      = sh_count_q;

    // temporaries defaults
    cmp_eq_full = 1'b0;
    cmp_lt_u    = 1'b0;
    cmp_lt_s    = 1'b0;
    cmp_take    = 1'b0;

    load32      = 32'h0;
    load_b8     = 8'h0;
    load_h16    = 16'h0;

    shamt_sel   = 5'd0;
    sh_nlo      = 16'h0;
    sh_nhi      = 16'h0;

    // ----------------------------------------------------------
    // FSM
    // ----------------------------------------------------------
    unique case (st_q)

      // ============================================================
      // FETCH 32-bit instruction as two 16-bit reads
      // ============================================================
      S_FETCH_LO_REQ: begin
        mem_valid  = 1'b1;
        mem_we     = 1'b0;
        mem_addr16 = pc_q[15:0];
        if (mem_ready) st_d = S_FETCH_LO_RESP;
      end

      S_FETCH_LO_RESP: begin
        ir_lo_we = 1'b1;
        pc_d     = pc_q + 32'd2;
        st_d     = S_FETCH_HI_REQ;
      end

      S_FETCH_HI_REQ: begin
        mem_valid  = 1'b1;
        mem_we     = 1'b0;
        mem_addr16 = pc_q[15:0];
        if (mem_ready) st_d = S_FETCH_HI_RESP;
      end

      S_FETCH_HI_RESP: begin
        ir_hi_we = 1'b1;
        pc_d     = pc_q + 32'd2;
        st_d     = S_DECODE;
      end

      // ============================================================
      // DECODE
      // ============================================================
      S_DECODE: begin
        // Most instructions need operands
        ab_latch_we = 1'b1;

        // Default: NOP
        kind_d    = K_NOP;
        op_d      = ALU_ADD;
        use_imm_d = 1'b0;
        wb_en_d   = 1'b0;

        // reset aux fields for this instruction
        mem_funct3_d    = funct3;
        br_funct3_d     = funct3;
        slt_unsigned_d  = 1'b0;

        sh_dir_d        = 1'b0;
        sh_arith_d      = 1'b0;
        sh_amt_is_imm_d = 1'b0;
        sh_count_d      = 5'd0;

        // LUI
        if (is_lui) begin
          if (!illegal_rd) begin
            kind_d  = K_LUI;
            wb_en_d = 1'b1;
            st_d    = S_WB;
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // AUIPC
        end else if (is_auipc) begin
          if (!illegal_rd) begin
            kind_d    = K_AUIPC;
            wb_en_d   = 1'b1;
            use_imm_d = 1'b1;
            op_d      = ALU_ADD;
            st_d      = S_EXEC_LO;
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // JAL
        end else if (is_jal) begin
          if (!illegal_rd) begin
            kind_d  = K_JAL;
            wb_en_d = 1'b1;
            st_d    = S_JAL_TGT_LO;
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // JALR
        end else if (is_jalr) begin
          if (!illegal_rs1 && !illegal_rd) begin
            kind_d  = K_JALR;
            wb_en_d = 1'b1;
            st_d    = S_ADDR_LO;
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // BRANCH
        end else if (is_branch) begin
          if (!illegal_rs1 && !illegal_rs2) begin
            kind_d      = K_BRANCH;
            br_funct3_d = funct3;
            st_d        = S_CMP_LO;
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // LOAD
        end else if (is_load) begin
          if (!illegal_rs1 && !illegal_rd) begin
            unique case (funct3)
              3'b000, 3'b001, 3'b010, 3'b100, 3'b101: begin
                kind_d       = K_LOAD;
                mem_funct3_d = funct3;
                wb_en_d      = 1'b1;
                st_d         = S_ADDR_LO;
              end
              default: st_d = S_FETCH_LO_REQ;
            endcase
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // STORE
        end else if (is_store) begin
          if (!illegal_rs1 && !illegal_rs2) begin
            unique case (funct3)
              3'b000, 3'b001, 3'b010: begin
                kind_d       = K_STORE;
                mem_funct3_d = funct3;
                wb_en_d      = 1'b0;
                st_d         = S_ADDR_LO;
              end
              default: st_d = S_FETCH_LO_REQ;
            endcase
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // OP (R-type)
        end else if (is_op) begin
          if (!illegal_rs1 && !illegal_rs2 && !illegal_rd) begin
            unique case (funct3)
              3'b000: begin
                kind_d  = K_ALU;
                wb_en_d = 1'b1;
                op_d    = (funct7[5] ? ALU_SUB : ALU_ADD);
                st_d    = S_EXEC_LO;
              end
              3'b111: begin kind_d=K_ALU; wb_en_d=1'b1; op_d=ALU_AND; st_d=S_EXEC_LO; end
              3'b110: begin kind_d=K_ALU; wb_en_d=1'b1; op_d=ALU_OR;  st_d=S_EXEC_LO; end
              3'b100: begin kind_d=K_ALU; wb_en_d=1'b1; op_d=ALU_XOR; st_d=S_EXEC_LO; end

              3'b010: begin // SLT
                kind_d         = K_SLT;
                wb_en_d        = 1'b1;
                use_imm_d      = 1'b0;
                slt_unsigned_d = 1'b0;
                st_d           = S_CMP_LO;
              end
              3'b011: begin // SLTU
                kind_d         = K_SLT;
                wb_en_d        = 1'b1;
                use_imm_d      = 1'b0;
                slt_unsigned_d = 1'b1;
                st_d           = S_CMP_LO;
              end

              3'b001: begin // SLL
                kind_d          = K_SHIFT;
                wb_en_d         = 1'b1;
                sh_dir_d        = 1'b0;
                sh_arith_d      = 1'b0;
                sh_amt_is_imm_d = 1'b0;
                st_d            = S_SHIFT_INIT;
              end
              3'b101: begin // SRL/SRA
                kind_d          = K_SHIFT;
                wb_en_d         = 1'b1;
                sh_dir_d        = 1'b1;
                sh_arith_d      = funct7[5];
                sh_amt_is_imm_d = 1'b0;
                st_d            = S_SHIFT_INIT;
              end

              default: st_d = S_FETCH_LO_REQ;
            endcase
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        // OP-IMM
        end else if (is_opimm) begin
          if (!illegal_rs1 && !illegal_rd) begin
            unique case (funct3)
              3'b000: begin kind_d=K_ALU; wb_en_d=1'b1; use_imm_d=1'b1; op_d=ALU_ADD; st_d=S_EXEC_LO; end
              3'b111: begin kind_d=K_ALU; wb_en_d=1'b1; use_imm_d=1'b1; op_d=ALU_AND; st_d=S_EXEC_LO; end
              3'b110: begin kind_d=K_ALU; wb_en_d=1'b1; use_imm_d=1'b1; op_d=ALU_OR;  st_d=S_EXEC_LO; end
              3'b100: begin kind_d=K_ALU; wb_en_d=1'b1; use_imm_d=1'b1; op_d=ALU_XOR; st_d=S_EXEC_LO; end

              3'b010: begin // SLTI
                kind_d         = K_SLT;
                wb_en_d        = 1'b1;
                use_imm_d      = 1'b1;
                slt_unsigned_d = 1'b0;
                st_d           = S_CMP_LO;
              end
              3'b011: begin // SLTIU
                kind_d         = K_SLT;
                wb_en_d        = 1'b1;
                use_imm_d      = 1'b1;
                slt_unsigned_d = 1'b1;
                st_d           = S_CMP_LO;
              end

              3'b001: begin // SLLI
                kind_d          = K_SHIFT;
                wb_en_d         = 1'b1;
                sh_dir_d        = 1'b0;
                sh_arith_d      = 1'b0;
                sh_amt_is_imm_d = 1'b1;
                st_d            = S_SHIFT_INIT;
              end
              3'b101: begin // SRLI/SRAI
                kind_d          = K_SHIFT;
                wb_en_d         = 1'b1;
                sh_dir_d        = 1'b1;
                sh_arith_d      = funct7[5];
                sh_amt_is_imm_d = 1'b1;
                st_d            = S_SHIFT_INIT;
              end

              default: st_d = S_FETCH_LO_REQ;
            endcase
          end else begin
            st_d = S_FETCH_LO_REQ;
          end

        end else begin
          st_d = S_FETCH_LO_REQ;
        end
      end

      // ============================================================
      // EXEC: 32-bit op over two 16-bit halves (ALU / AUIPC)
      // ============================================================
      S_EXEC_LO: begin
        alu_op = op_q;

        // A source: AUIPC uses PC(inst), otherwise rs1
        alu_a_sel = (kind_q == K_AUIPC) ? 3'd4 : 3'd0;

        // B source: imm vs rs2
        alu_b_sel = use_imm_q ? 3'd3 : 3'd0;

        r_lo_we  = 1'b1;
        carry_we = (op_q == ALU_ADD) || (op_q == ALU_SUB);

        st_d = S_EXEC_HI;
      end

      S_EXEC_HI: begin
        alu_op = op_q;

        alu_a_sel = (kind_q == K_AUIPC) ? 3'd5 : 3'd1;

        alu_b_sel = use_imm_q ? 3'd4 : 3'd1;

        alu_use_carry_in = (op_q == ALU_ADD) || (op_q == ALU_SUB);

        r_hi_we = 1'b1;

        st_d = S_WB;
      end

      // ============================================================
      // WB: commit ALU/SHIFT/SLT/LUI/AUIPC results
      // ============================================================
      S_WB: begin
        if (wb_en_q) begin
          rf_we    = 1'b1;
          rf_waddr = rd;

          if (kind_q == K_LUI) begin
            rf_wdata_lo = imm_lo;
            rf_wdata_hi = imm_hi;
          end else begin
            rf_wdata_lo = r_lo;
            rf_wdata_hi = r_hi;
          end
        end
        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // Address generation: addr = rs1 + imm (split add)
      // Used for LOAD/STORE/JALR
      // ============================================================
      S_ADDR_LO: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd0; // A_lo
        alu_b_sel = 3'd3; // imm_lo

        addr_lo_we = 1'b1;
        carry_we   = 1'b1;

        st_d = S_ADDR_HI;
      end

      S_ADDR_HI: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd1; // A_hi
        alu_b_sel = 3'd4; // imm_hi
        alu_use_carry_in = 1'b1;

        addr_hi_we = 1'b1;

        if (kind_q == K_LOAD) begin
          st_d = S_MEM_RD_LO_REQ;
        end else if (kind_q == K_STORE) begin
          st_d = (mem_funct3_q == 3'b000) ? S_SB_RD_REQ : S_MEM_WR_LO_REQ;
        end else if (kind_q == K_JALR) begin
          st_d = S_WB_JUMP;
        end else begin
          st_d = S_FETCH_LO_REQ;
        end
      end

      // ============================================================
      // LOAD
      // ============================================================
      S_MEM_RD_LO_REQ: begin
        mem_valid  = 1'b1;
        mem_we     = 1'b0;
        mem_addr16 = addr_lo_q;
        if (mem_ready) st_d = S_MEM_RD_LO_RESP;
      end

      S_MEM_RD_LO_RESP: begin
        mem_buf_lo_we = 1'b1;
        st_d = (mem_funct3_q == 3'b010) ? S_MEM_RD_HI_REQ : S_WB_LOAD;
      end

      S_MEM_RD_HI_REQ: begin
        mem_valid  = 1'b1;
        mem_we     = 1'b0;
        mem_addr16 = addr_lo_q + 16'd2;
        if (mem_ready) st_d = S_MEM_RD_HI_RESP;
      end

      S_MEM_RD_HI_RESP: begin
        mem_buf_hi_we = 1'b1;
        st_d = S_WB_LOAD;
      end

      S_WB_LOAD: begin
        load_b8  = addr_lo_q[0] ? mem_buf_lo[15:8] : mem_buf_lo[7:0];
        load_h16 = mem_buf_lo;

        unique case (mem_funct3_q)
          3'b000: load32 = {{24{load_b8[7]}}, load_b8};          // LB
          3'b100: load32 = {24'h0, load_b8};                     // LBU
          3'b001: load32 = {{16{load_h16[15]}}, load_h16};       // LH
          3'b101: load32 = {16'h0, load_h16};                    // LHU
          3'b010: load32 = {mem_buf_hi, mem_buf_lo};             // LW
          default: load32 = 32'h0;
        endcase

        rf_we       = 1'b1;
        rf_waddr    = rd;
        rf_wdata_lo = load32[15:0];
        rf_wdata_hi = load32[31:16];

        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // STORE (SH/SW). SB handled with RMW states below.
      // ============================================================
      S_MEM_WR_LO_REQ: begin
        mem_valid   = 1'b1;
        mem_we      = 1'b1;
        mem_addr16  = addr_lo_q;
        mem_wdata16 = b_lo; // SH/SW

        if (mem_ready) st_d = S_MEM_WR_LO_RESP;
      end

      S_MEM_WR_LO_RESP: begin
        st_d = (mem_funct3_q == 3'b010) ? S_MEM_WR_HI_REQ : S_FETCH_LO_REQ;
      end

      S_MEM_WR_HI_REQ: begin
        mem_valid   = 1'b1;
        mem_we      = 1'b1;
        mem_addr16  = addr_lo_q + 16'd2;
        mem_wdata16 = b_hi;
        if (mem_ready) st_d = S_MEM_WR_HI_RESP;
      end

      S_MEM_WR_HI_RESP: begin
        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // SB (store byte) read-modify-write
      // ============================================================
      S_SB_RD_REQ: begin
        mem_valid  = 1'b1;
        mem_we     = 1'b0;
        mem_addr16 = addr_lo_q;
        if (mem_ready) st_d = S_SB_RD_RESP;
      end

      S_SB_RD_RESP: begin
        mem_buf_lo_we = 1'b1;
        st_d = S_SB_WR_REQ;
      end

      S_SB_WR_REQ: begin
        mem_valid   = 1'b1;
        mem_we      = 1'b1;
        mem_addr16  = addr_lo_q;
        mem_wdata16 = addr_lo_q[0] ? {b_lo[7:0], mem_buf_lo[7:0]} : {mem_buf_lo[15:8], b_lo[7:0]};
        if (mem_ready) st_d = S_SB_WR_RESP;
      end

      S_SB_WR_RESP: begin
        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // CMP (used for BRANCH and SLT*)
      // ============================================================
      S_CMP_LO: begin
        alu_op    = ALU_SUB;
        alu_a_sel = 3'd0;
        alu_b_sel = use_imm_q ? 3'd3 : 3'd0;

        eq_lo_d   = alu_z;
        carry_we  = 1'b1; // latch borrow for high half

        st_d = S_CMP_HI;
      end

      S_CMP_HI: begin
        alu_op    = ALU_SUB;
        alu_a_sel = 3'd1;
        alu_b_sel = use_imm_q ? 3'd4 : 3'd1;
        alu_use_carry_in = 1'b1;

        cmp_eq_full = eq_lo_q && alu_z;
        cmp_lt_u    = alu_cout; // borrow-out indicates a<b unsigned

        // signed lt
        if (a_hi[15] != (use_imm_q ? imm_hi[15] : b_hi[15])) cmp_lt_s = a_hi[15];
        else                                                 cmp_lt_s = alu_n;

        if (kind_q == K_BRANCH) begin
          unique case (br_funct3_q)
            3'b000: cmp_take =  cmp_eq_full; // BEQ
            3'b001: cmp_take = !cmp_eq_full; // BNE
            3'b100: cmp_take =  cmp_lt_s;    // BLT
            3'b101: cmp_take = !cmp_lt_s;    // BGE
            3'b110: cmp_take =  cmp_lt_u;    // BLTU
            3'b111: cmp_take = !cmp_lt_u;    // BGEU
            default: cmp_take = 1'b0;
          endcase

          st_d = cmp_take ? S_BR_TGT_LO : S_FETCH_LO_REQ;

        end else if (kind_q == K_SLT) begin
          r_use_ext = 1'b1;
          r_ext_lo  = slt_unsigned_q ? (cmp_lt_u ? 16'd1 : 16'd0)
                                     : (cmp_lt_s ? 16'd1 : 16'd0);
          r_ext_hi  = 16'h0;
          r_lo_we   = 1'b1;
          r_hi_we   = 1'b1;
          st_d      = S_WB;

        end else begin
          st_d = S_FETCH_LO_REQ;
        end
      end

      // ============================================================
      // Branch target: PC(inst) + imm
      // ============================================================
      S_BR_TGT_LO: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd4; // PCINST_LO
        alu_b_sel = 3'd3; // imm_lo

        addr_lo_we = 1'b1;
        carry_we   = 1'b1;

        st_d = S_BR_TGT_HI;
      end

      S_BR_TGT_HI: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd5; // PCINST_HI
        alu_b_sel = 3'd4; // imm_hi
        alu_use_carry_in = 1'b1;

        addr_hi_we = 1'b1;

        pc_d = {alu_y, addr_lo_q};
        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // JAL target: PC(inst) + imm (compute first, then WB_JUMP applies redirect)
      // ============================================================
      S_JAL_TGT_LO: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd4; // PCINST_LO
        alu_b_sel = 3'd3; // imm_lo

        addr_lo_we = 1'b1;
        carry_we   = 1'b1;

        st_d = S_JAL_TGT_HI;
      end

      S_JAL_TGT_HI: begin
        alu_op    = ALU_ADD;
        alu_a_sel = 3'd5; // PCINST_HI
        alu_b_sel = 3'd4; // imm_hi
        alu_use_carry_in = 1'b1;

        addr_hi_we = 1'b1;

        st_d = S_WB_JUMP;
      end

      // ============================================================
      // WB_JUMP: write rd = PC+4 (pc_q), then redirect PC
      // ============================================================
      S_WB_JUMP: begin
        if (wb_en_q) begin
          rf_we       = 1'b1;
          rf_waddr    = rd;
          rf_wdata_lo = pc_q[15:0];
          rf_wdata_hi = pc_q[31:16];
        end

        if (kind_q == K_JAL) begin
          pc_d = {addr_hi_q, addr_lo_q};
        end else if (kind_q == K_JALR) begin
          pc_d = {addr_hi_q, (addr_lo_q & 16'hFFFE)};
        end else begin
          pc_d = pc_q;
        end

        st_d = S_FETCH_LO_REQ;
      end

      // ============================================================
      // Serial Shifter (multi-cycle)
      // ============================================================
      S_SHIFT_INIT: begin
        // Load operand into r regs
        r_use_ext = 1'b1;
        r_ext_lo  = a_lo;
        r_ext_hi  = a_hi;
        r_lo_we   = 1'b1;
        r_hi_we   = 1'b1;

        // Choose amount
        shamt_sel  = sh_amt_is_imm_q ? shamt5 : b_lo[4:0];
        sh_count_d = shamt_sel;

        st_d = (shamt_sel == 5'd0) ? S_WB : S_SHIFT_STEP;
      end

      S_SHIFT_STEP: begin
        // shift by 1 bit each cycle, crossing halves
        if (!sh_dir_q) begin
          sh_nlo = {r_lo[14:0], 1'b0};
          sh_nhi = {r_hi[14:0], r_lo[15]};
        end else begin
          sh_nlo = {r_hi[0], r_lo[15:1]};
          sh_nhi = sh_arith_q ? {r_hi[15], r_hi[15:1]} : {1'b0, r_hi[15:1]};
        end

        r_use_ext = 1'b1;
        r_ext_lo  = sh_nlo;
        r_ext_hi  = sh_nhi;
        r_lo_we   = 1'b1;
        r_hi_we   = 1'b1;

        if (sh_count_q != 0) sh_count_d = sh_count_q - 5'd1;

        st_d = (sh_count_q == 5'd1) ? S_WB : S_SHIFT_STEP;
      end

      default: begin
        st_d = S_FETCH_LO_REQ;
      end

    endcase
  end

endmodule

// ================================================================
// 3) RV32E regfile split: 16 x 32-bit as 2x(16 x 16-bit)
// ================================================================
module regfile_rv32e_split (
  input  logic        clk,
  input  logic        rst_n,

  input  logic [3:0]  raddr1,
  output logic [15:0] rdata1_lo,
  output logic [15:0] rdata1_hi,

  input  logic [3:0]  raddr2,
  output logic [15:0] rdata2_lo,
  output logic [15:0] rdata2_hi,

  input  logic        we,
  input  logic [3:0]  waddr,
  input  logic [15:0] wdata_lo,
  input  logic [15:0] wdata_hi
);
  logic [15:0] rf_lo [0:15];
  logic [15:0] rf_hi [0:15];

  // combinational reads
  always_comb begin
    rdata1_lo = (raddr1 == 4'd0) ? 16'h0 : rf_lo[raddr1];
    rdata1_hi = (raddr1 == 4'd0) ? 16'h0 : rf_hi[raddr1];
    rdata2_lo = (raddr2 == 4'd0) ? 16'h0 : rf_lo[raddr2];
    rdata2_hi = (raddr2 == 4'd0) ? 16'h0 : rf_hi[raddr2];
  end

  // synchronous write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin : init_rf
      integer i;
      for (i = 0; i < 16; i++) begin
        rf_lo[i] <= 16'h0;
        rf_hi[i] <= 16'h0;
      end
    end else begin
      if (we && (waddr != 4'd0)) begin
        rf_lo[waddr] <= wdata_lo;
        rf_hi[waddr] <= wdata_hi;
      end
      // keep x0 forced to 0
      rf_lo[0] <= 16'h0;
      rf_hi[0] <= 16'h0;
    end
  end

endmodule


// ================================================================
// 4) 16-bit ALU
// ================================================================
module alu16 (
  input  logic [15:0] a,
  input  logic [15:0] b,
  input  aluop_t      op,
  input  logic        cin,   // carry-in for ADD, borrow-in for SUB
  output logic [15:0] y,
  output logic        cout,  // carry-out for ADD, borrow-out for SUB
  output logic        z,
  output logic        n
);

  logic [16:0] tmp;

  always_comb begin
    y    = 16'h0;
    cout = 1'b0;
    tmp  = 17'h0;

    unique case (op)
      ALU_ADD: begin
        tmp  = {1'b0, a} + {1'b0, b} + {16'h0, cin};
        y    = tmp[15:0];
        cout = tmp[16];            // carry
      end
      ALU_SUB: begin
        // Unsigned subtract with borrow chain:
        // borrow-out is tmp[16] (1 when underflow occurred)
        tmp  = {1'b0, a} - {1'b0, b} - {16'h0, cin};
        y    = tmp[15:0];
        cout = tmp[16];            // borrow
      end
      ALU_AND:  y = a & b;
      ALU_OR :  y = a | b;
      ALU_XOR:  y = a ^ b;
      ALU_PASS: y = a;
      default:  y = 16'h0;
    endcase
  end

  assign z = (y == 16'h0);
  assign n = y[15];

endmodule

// ================================================================
// 5) Memory / TT pin adapter (skeleton)
// ================================================================
// IMPORTANT: Your exact TT template may differ. This is a *placeholder*
// adapter that shows typical mapping patterns.
// You must decide your external memory signaling protocol.
//
// Here’s a minimal approach:
// - Drive address on uo_out/uio_out when mem_valid=1
// - Drive write data on the same bus when mem_we=1 (time-mux via phase)
// - Read data comes in on ui_in/uio_in when mem_ready=1
//
// If you have only one 16-bit outbound bus, you need a "phase" concept.
// You can implement phase inside adapter (toggle on handshake) or drive it
// from the controller (preferred).
// ================================================================
module mem_if_adapter_tt (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        mem_valid,
  output logic        mem_ready,
  input  logic        mem_we,
  input  logic [15:0] mem_addr16,    // byte address
  input  logic [15:0] mem_wdata16,
  output logic [15:0] mem_rdata16,

  input  logic [7:0]  ui_in,         // data in [7:0]
  output logic [7:0]  uo_out,        // addr/data out [7:0]
  input  logic [7:0]  uio_in,        // data in [15:8] (when uio_oe=0)
  output logic [7:0]  uio_out,       // addr/data out [15:8] (when uio_oe=1)
  output logic [7:0]  uio_oe         // 1=drive uio_out, 0=sample uio_in
);

  // ------------------------------------------------------------
  // Tiny Tapeout 16-bit "time-mux" memory protocol (2-cycle):
  //   Cycle 0 (ADDR phase): core drives address on {uio_out,uo_out}
  //                         uio_oe=1 (drive upper byte)
  //   Cycle 1 (DATA phase):
  //     - READ: core releases uio pins (uio_oe=0) and samples
  //             {uio_in,ui_in} into mem_rdata16_hold
  //     - WRITE: core drives data on {uio_out,uo_out}, uio_oe=1
  //   mem_ready pulses high for 1 cycle when DATA phase completes.
  //
  // Notes:
  // - ui_in are always inputs, so external memory may drive them any time.
  // - The external memory/controller must latch the address during ADDR.
  // ------------------------------------------------------------

  typedef enum logic [1:0] { M_IDLE=2'd0, M_ADDR=2'd1, M_DATA=2'd2 } mstate_t;
  mstate_t ms_q, ms_d;

  // Latched request (so CPU can hold mem_valid without glitching signals)
  logic        we_q, we_d;
  logic [15:0] addr_q, addr_d;
  logic [15:0] wdata_q, wdata_d;

  logic [15:0] rdata_hold_q, rdata_hold_d;
  logic        ready_pulse;

  // Output word and direction for this cycle
  logic [15:0] out_word16;
  logic        drive_uio;  // 1=output mode, 0=input mode

  // Default readback
  assign mem_rdata16 = rdata_hold_q;

  // Sequential
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ms_q         <= M_IDLE;
      we_q         <= 1'b0;
      addr_q       <= 16'h0;
      wdata_q      <= 16'h0;
      rdata_hold_q <= 16'h0;
    end else begin
      ms_q         <= ms_d;
      we_q         <= we_d;
      addr_q       <= addr_d;
      wdata_q      <= wdata_d;
      rdata_hold_q <= rdata_hold_d;
    end
  end

  // Next-state / latching
  always_comb begin
    ms_d         = ms_q;
    we_d         = we_q;
    addr_d       = addr_q;
    wdata_d      = wdata_q;
    rdata_hold_d = rdata_hold_q;

    ready_pulse  = 1'b0;

    unique case (ms_q)
      M_IDLE: begin
        if (mem_valid) begin
          // capture request at start
          we_d    = mem_we;
          addr_d  = mem_addr16;
          wdata_d = mem_wdata16;
          ms_d    = M_ADDR;
        end
      end

      M_ADDR: begin
        // one full cycle of address drive
        ms_d = M_DATA;
      end

      M_DATA: begin
        // complete transaction
        if (!we_q) begin
          // READ: sample data bus in DATA phase
          rdata_hold_d = {uio_in, ui_in};
        end
        ready_pulse = 1'b1;
        // require mem_valid deassert before accepting a new request
        if (!mem_valid) ms_d = M_IDLE;
        else            ms_d = M_IDLE; // simplest: accept next only after core drops mem_valid
      end

      default: ms_d = M_IDLE;
    endcase
  end

  // Drive outputs based on state
  always_comb begin
    mem_ready = ready_pulse;

    // defaults
    out_word16 = 16'h0;
    drive_uio  = 1'b0;

    unique case (ms_q)
      M_ADDR: begin
        out_word16 = addr_q;
        drive_uio  = 1'b1;
      end
      M_DATA: begin
        if (we_q) begin
          // WRITE: drive write data
          out_word16 = wdata_q;
          drive_uio  = 1'b1;
        end else begin
          // READ: release uio pins so external can drive upper byte
          out_word16 = 16'h0;
          drive_uio  = 1'b0;
        end
      end
      default: begin
        out_word16 = 16'h0;
        drive_uio  = 1'b0;
      end
    endcase

    uo_out  = out_word16[7:0];
    uio_out = out_word16[15:8];
    uio_oe  = drive_uio ? 8'hFF : 8'h00;
  end

endmodule

// ================================================================
// 6) Optional: simple memory model for simulation (B2 capable)
// ================================================================
// Drop this in your testbench, not on silicon.
// You can connect CPU.mem_* to this instead of the TT adapter.
// ================================================================
module mem16_model #(
  parameter int LATENCY = 1
) (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        mem_valid,
  output logic        mem_ready,
  input  logic        mem_we,
  input  logic [15:0] mem_addr16,
  input  logic [15:0] mem_wdata16,
  output logic [15:0] mem_rdata16
);

  logic [15:0] mem [0:32767]; // 64KB bytes => 32K halfwords (since 16-bit accesses)

  // latency pipeline
  logic [$clog2(LATENCY+1)-1:0] cnt_q, cnt_d;
  logic pending_q, pending_d;
  logic we_q;
  logic [15:0] addr_q, wdata_q;

  function automatic [14:0] hw_addr(input logic [15:0] byte_addr);
    hw_addr = byte_addr[15:1]; // halfword index (assumes aligned)
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt_q <= '0;
      pending_q <= 1'b0;
      we_q <= 1'b0;
      addr_q <= 16'h0;
      wdata_q <= 16'h0;
    end else begin
      cnt_q <= cnt_d;
      pending_q <= pending_d;
      if (mem_valid && !pending_q) begin
        we_q    <= mem_we;
        addr_q  <= mem_addr16;
        wdata_q <= mem_wdata16;
      end
      if (mem_ready && pending_q && we_q) begin
        mem[hw_addr(addr_q)] <= wdata_q;
      end
    end
  end

  always_comb begin
    pending_d = pending_q;
    cnt_d = cnt_q;
    mem_ready = 1'b0;

    if (!pending_q) begin
      if (mem_valid) begin
        pending_d = 1'b1;
        cnt_d = LATENCY[$bits(cnt_q)-1:0];
      end
    end else begin
      if (cnt_q == 0) begin
        mem_ready = 1'b1;
        pending_d = 1'b0;
      end else begin
        cnt_d = cnt_q - 1;
      end
    end
  end

  always_comb begin
    mem_rdata16 = mem[hw_addr(addr_q)];
  end

endmodule