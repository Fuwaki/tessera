//! Real datapath pieces for the CIRCT hybrid path: requant, ROMs, integer
//! schedule core, and top-level wiring helpers.
//!
//! The schedule core implements the same integer formula as
//! [`super::circt_data::evaluate_integer`] so Verilator can bit-exact check.

use std::fmt::Write as _;

use crate::dialect::read::OpKind as DopKind;

use super::circt_data::ResolvedOp;
use super::{RtlError, bits_for};

/// Comb requant: `(acc * m0 + round) >>> shift` then clamp to [lo, hi].
/// `shift` is a compile-time constant baked into the module name / body.
pub fn emit_requant_const(ir: &mut String, shift: u32) {
    let name = format!("tessera_requant_s{shift}");
    if ir.contains(&format!("hw.module @{name}")) {
        return;
    }
    let _ = writeln!(ir, "// requant shift={shift}");
    let _ = writeln!(
        ir,
        "hw.module @{name}(in %acc : i32, in %m0 : i32, in %lo : i16, in %hi : i16, out o : i16) {{"
    );
    // Sign-extend acc and m0 to i64 for the product.
    let _ = writeln!(ir, "  %acc_s = comb.extract %acc from 31 : (i32) -> i1");
    let _ = writeln!(ir, "  %acc_r = comb.replicate %acc_s : (i1) -> i32");
    let _ = writeln!(ir, "  %acc64 = comb.concat %acc_r, %acc : i32, i32");
    let _ = writeln!(ir, "  %m0_s = comb.extract %m0 from 31 : (i32) -> i1");
    let _ = writeln!(ir, "  %m0_r = comb.replicate %m0_s : (i1) -> i32");
    let _ = writeln!(ir, "  %m064 = comb.concat %m0_r, %m0 : i32, i32");
    let _ = writeln!(ir, "  %prod = comb.mul %acc64, %m064 : i64");
    if shift == 0 {
        let _ = writeln!(ir, "  %shifted = comb.extract %prod from 0 : (i64) -> i32");
    } else {
        let round_pos = 1u64 << (shift - 1);
        let round_neg = round_pos - 1;
        let _ = writeln!(ir, "  %c0i64 = hw.constant 0 : i64");
        let _ = writeln!(ir, "  %prod_neg = comb.icmp bin slt %prod, %c0i64 : i64");
        let _ = writeln!(ir, "  %round_pos = hw.constant {round_pos} : i64");
        let _ = writeln!(ir, "  %round_neg = hw.constant {round_neg} : i64");
        let _ = writeln!(
            ir,
            "  %round = comb.mux %prod_neg, %round_neg, %round_pos : i64"
        );
        let _ = writeln!(ir, "  %biased = comb.add %prod, %round : i64");
        // Arithmetic shift via extract of the high part after the shift.
        // For signed >>> shift: take bits [shift+31 : shift].
        let _ = writeln!(
            ir,
            "  %shifted = comb.extract %biased from {shift} : (i64) -> i32"
        );
    }
    // Clamp shifted (i32) into [lo, hi] as i16.
    let _ = writeln!(ir, "  %lo_s = comb.extract %lo from 15 : (i16) -> i1");
    let _ = writeln!(ir, "  %lo_r = comb.replicate %lo_s : (i1) -> i16");
    let _ = writeln!(ir, "  %lo32 = comb.concat %lo_r, %lo : i16, i16");
    let _ = writeln!(ir, "  %hi_s = comb.extract %hi from 15 : (i16) -> i1");
    let _ = writeln!(ir, "  %hi_r = comb.replicate %hi_s : (i1) -> i16");
    let _ = writeln!(ir, "  %hi32 = comb.concat %hi_r, %hi : i16, i16");
    let _ = writeln!(ir, "  %lt = comb.icmp bin slt %shifted, %lo32 : i32");
    let _ = writeln!(ir, "  %gt = comb.icmp bin sgt %shifted, %hi32 : i32");
    let _ = writeln!(ir, "  %clamped_lo = comb.mux %lt, %lo32, %shifted : i32");
    let _ = writeln!(ir, "  %clamped = comb.mux %gt, %hi32, %clamped_lo : i32");
    let _ = writeln!(ir, "  %o = comb.extract %clamped from 0 : (i32) -> i16");
    let _ = writeln!(ir, "  hw.output %o : i16");
    let _ = writeln!(ir, "}}");
    let _ = writeln!(ir);
}

/// Emit an i8 ROM as `hw.array_create` + `hw.array_get`.
pub fn emit_i8_rom(ir: &mut String, name: &str, data: &[i8]) {
    if ir.contains(&format!("hw.module @{name}")) {
        return;
    }
    let n = data.len().max(1);
    let aw = bits_for(n as u32).max(1);
    let _ = writeln!(ir, "// i8 ROM {name} ({n} entries)");
    let _ = writeln!(ir, "hw.module @{name}(in %addr : i{aw}, out data : i8) {{");
    if data.is_empty() {
        let _ = writeln!(ir, "  %z = hw.constant 0 : i8");
        let _ = writeln!(ir, "  hw.output %z : i8");
    } else {
        for (i, &v) in data.iter().enumerate() {
            let _ = writeln!(ir, "  %e{i} = hw.constant {v} : i8");
        }
        let elems = (0..data.len()).rev().map(|i| format!("%e{i}")).collect::<Vec<_>>().join(", ");
        let _ = writeln!(ir, "  %arr = hw.array_create {elems} : i8");
        // hw.array_get expects index width = log2(ceil_pow2(n)) sometimes;
        // cast addr to the array index type CIRCT expects (same bits_for).
        let _ = writeln!(ir, "  %data = hw.array_get %arr[%addr] : !hw.array<{n}xi8>, i{aw}");
        let _ = writeln!(ir, "  hw.output %data : i8");
    }
    let _ = writeln!(ir, "}}");
    let _ = writeln!(ir);
}

pub fn emit_i32_rom(ir: &mut String, name: &str, data: &[i32]) {
    if ir.contains(&format!("hw.module @{name}")) {
        return;
    }
    let n = data.len().max(1);
    let aw = bits_for(n as u32).max(1);
    let _ = writeln!(ir, "// i32 ROM {name} ({n} entries)");
    let _ = writeln!(ir, "hw.module @{name}(in %addr : i{aw}, out data : i32) {{");
    if data.is_empty() {
        let _ = writeln!(ir, "  %z = hw.constant 0 : i32");
        let _ = writeln!(ir, "  hw.output %z : i32");
    } else {
        for (i, &v) in data.iter().enumerate() {
            let _ = writeln!(ir, "  %e{i} = hw.constant {v} : i32");
        }
        let elems = (0..data.len()).rev().map(|i| format!("%e{i}")).collect::<Vec<_>>().join(", ");
        let _ = writeln!(ir, "  %arr = hw.array_create {elems} : i32");
        let _ = writeln!(ir, "  %data = hw.array_get %arr[%addr] : !hw.array<{n}xi32>, i{aw}");
        let _ = writeln!(ir, "  hw.output %data : i32");
    }
    let _ = writeln!(ir, "}}");
    let _ = writeln!(ir);
}

/// Slot layout for the shared activation memory inside the schedule core.
#[derive(Debug, Clone)]
pub struct MemLayout {
    pub total_words: u64,
    /// base address for graph input
    pub input_base: u64,
    pub input_words: u64,
    /// op_index -> base
    pub op_base: Vec<(u32, u64, u64)>,
}

impl MemLayout {
    pub fn from_ops(ops: &[ResolvedOp], in_shape: [u32; 3]) -> Self {
        let (ih, iw, ic) = (in_shape[0], in_shape[1], in_shape[2]);
        let input_words = u64::from(ic) * u64::from(ih) * u64::from(iw);
        let mut cursor = input_words;
        let mut op_base = Vec::new();
        for op in ops {
            let words = op.out_words().max(1);
            op_base.push((op.index, cursor, words));
            cursor += words;
        }
        Self {
            total_words: cursor.max(1),
            input_base: 0,
            input_words: input_words.max(1),
            op_base,
        }
    }

    pub fn base_of(&self, src: Option<u32>) -> u64 {
        match src {
            None => self.input_base,
            Some(idx) => self
                .op_base
                .iter()
                .find(|(i, _, _)| *i == idx)
                .map(|(_, b, _)| *b)
                .unwrap_or(0),
        }
    }
}

/// Emit the integer schedule core: load input → run every ResolvedOp → expose output.
///
/// This is the functionally-correct datapath. Engines/arenas in the top remain
/// for structural hybrid wiring; the core is what the bit-exact TB checks.
pub fn emit_sched_core(
    ir: &mut String,
    ops: &[ResolvedOp],
    in_shape: [u32; 3],
    results: &[u32],
    // Parallel to ops: which hybrid engine index owns that op (from binds).
    op_engines: &[usize],
    n_engines: usize,
    dut_name: &str,
) -> Result<MemLayout, RtlError> {
    if ops.is_empty() {
        return Err(RtlError::UnsupportedLayout {
            detail: "sched core needs resolved ops".into(),
        });
    }
    let layout = MemLayout::from_ops(ops, in_shape);
    let words = layout.total_words;
    let aw = bits_for(words as u32).max(1);
    let name = format!("{dut_name}_sched_core");

    // ROMs for every conv-like op.
    for op in ops {
        if !op.weights.is_empty() {
            emit_i8_rom(ir, &format!("{name}_w{0}", op.index), &op.weights);
        }
        if !op.bias.is_empty() {
            emit_i32_rom(ir, &format!("{name}_b{0}", op.index), &op.bias);
        }
        if !op.m0.is_empty() {
            let m0_i32: Vec<i32> = op
                .m0
                .iter()
                .map(|&v| i32::try_from(v).unwrap_or(v as i32))
                .collect();
            emit_i32_rom(ir, &format!("{name}_m{0}", op.index), &m0_i32);
            emit_requant_const(ir, op.shift);
        }
    }
    // Default requant s0 for add-style.
    emit_requant_const(ir, 0);

    let out_idx = results.first().copied().unwrap_or(ops.last().unwrap().index);
    let out_base = layout.base_of(Some(out_idx));
    let out_words = layout
        .op_base
        .iter()
        .find(|(i, _, _)| *i == out_idx)
        .map(|(_, _, w)| *w)
        .unwrap_or(1);
    let out_aw = bits_for(out_words.max(1) as u32).max(1);
    let in_aw = bits_for(layout.input_words.max(1) as u32).max(1);

    // Host ports use input-sized and output-sized addresses; internally we use full aw.
    let _ = writeln!(
        ir,
        "// integer schedule core — bit-exact vs circt_data::evaluate_integer"
    );
    let neng = n_engines.max(1);
    let busy_ty = if neng == 1 { "i1".into() } else { format!("i{neng}") };
    let _ = writeln!(
        ir,
        "hw.module @{name}(in %clk : i1, in %rst : i1, in %start : i1, out done : i1, out busy : i1, out engines_busy : {busy_ty}, in %ld_en : i1, in %ld_addr : i{in_aw}, in %ld_data : i16, in %rd_addr : i{out_aw}, out rd_data : i16) {{"
    );

    for (n, t, v) in [
        ("c0i1", "i1", "0"),
        ("c1i1", "i1", "1"),
        ("c0i2", "i2", "0"),
        ("c1i2", "i2", "1"),
        ("c2i2", "i2", "2"),
        ("c3i2", "i2", "3"),
        ("c0i16", "i16", "0"),
        ("c0i32", "i32", "0"),
        ("c0i8", "i8", "0"),
    ] {
        let _ = writeln!(ir, "  %{n} = hw.constant {v} : {t}");
    }
    let _ = writeln!(ir, "  %c0aw = hw.constant 0 : i{aw}");
    let _ = writeln!(ir, "  %c1aw = hw.constant 1 : i{aw}");

    // Activation memory.
    let _ = writeln!(ir, "  %mem = sv.reg : !hw.inout<uarray<{words}xi16>>");

    // Control regs.
    let _ = writeln!(ir, "  %st = sv.reg : !hw.inout<i2>");
    let _ = writeln!(ir, "  %done_r = sv.reg : !hw.inout<i1>");
    let _ = writeln!(ir, "  %busy_r = sv.reg : !hw.inout<i1>");
    let nop = ops.len().max(1);
    let opb = bits_for(nop as u32).max(1);
    let _ = writeln!(ir, "  %op_r = sv.reg : !hw.inout<i{opb}>");
    // Pixel / mac counters — wide enough for worst op.
    let max_pix = ops
        .iter()
        .map(|o| o.out_words())
        .max()
        .unwrap_or(1)
        .max(1);
    let max_inner = ops
        .iter()
        .map(|o| {
            let icg = (o.ic / o.groups.max(1)).max(1);
            u64::from(icg) * u64::from(o.kernel[0].max(1)) * u64::from(o.kernel[1].max(1))
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let pb = bits_for(max_pix as u32).max(1);
    let ib = bits_for(max_inner as u32).max(1);
    let _ = writeln!(ir, "  %pix = sv.reg : !hw.inout<i{pb}>");
    let _ = writeln!(ir, "  %inner = sv.reg : !hw.inout<i{ib}>");
    let _ = writeln!(ir, "  %acc_r = sv.reg : !hw.inout<i32>");
    let _ = writeln!(ir, "  %phase = sv.reg : !hw.inout<i2>"); // 0=init pixel, 1=mac, 2=write

    let _ = writeln!(ir, "  %stv = sv.read_inout %st : !hw.inout<i2>");
    let _ = writeln!(ir, "  %donev = sv.read_inout %done_r : !hw.inout<i1>");
    let _ = writeln!(ir, "  %busyv = sv.read_inout %busy_r : !hw.inout<i1>");
    let _ = writeln!(ir, "  %opv = sv.read_inout %op_r : !hw.inout<i{opb}>");
    let _ = writeln!(ir, "  %pixv = sv.read_inout %pix : !hw.inout<i{pb}>");
    let _ = writeln!(ir, "  %innerv = sv.read_inout %inner : !hw.inout<i{ib}>");
    let _ = writeln!(ir, "  %accv = sv.read_inout %acc_r : !hw.inout<i32>");
    let _ = writeln!(ir, "  %phasev = sv.read_inout %phase : !hw.inout<i2>");

    // Map host ld_addr (input-local) to absolute.
    cast_zext(ir, "ld_abs", "ld_addr", in_aw, aw);
    let _ = writeln!(ir, "  %ld_ptr = sv.array_index_inout %mem[%ld_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}");

    // Map host rd_addr (output-local) to absolute out_base+rd_addr.
    cast_zext(ir, "rd_local", "rd_addr", out_aw, aw);
    let _ = writeln!(ir, "  %out_base_c = hw.constant {out_base} : i{aw}");
    let _ = writeln!(ir, "  %rd_abs = comb.add %out_base_c, %rd_local : i{aw}");
    let _ = writeln!(ir, "  %rd_ptr = sv.array_index_inout %mem[%rd_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}");
    let _ = writeln!(ir, "  %rd_data_v = sv.read_inout %rd_ptr : !hw.inout<i16>");

    // Precompute per-op constants as mux trees selected by opv.
    emit_op_mux_trees(ir, ops, &layout, opb, pb, ib, aw)?;

    let _ = writeln!(ir, "  %is_idle = comb.icmp bin eq %stv, %c0i2 : i2");
    let _ = writeln!(ir, "  %is_run = comb.icmp bin eq %stv, %c1i2 : i2");
    let _ = writeln!(ir, "  %fire = comb.and %is_idle, %start : i1");
    let _ = writeln!(ir, "  %c0op = hw.constant 0 : i{opb}");
    let _ = writeln!(ir, "  %c1op = hw.constant 1 : i{opb}");
    let _ = writeln!(ir, "  %last_op = hw.constant {} : i{opb}", ops.len() - 1);
    let _ = writeln!(ir, "  %c0pix = hw.constant 0 : i{pb}");
    let _ = writeln!(ir, "  %c1pix = hw.constant 1 : i{pb}");
    let _ = writeln!(ir, "  %c0in = hw.constant 0 : i{ib}");
    let _ = writeln!(ir, "  %c1in = hw.constant 1 : i{ib}");
    let _ = writeln!(ir, "  %ph0 = hw.constant 0 : i2");
    let _ = writeln!(ir, "  %ph1 = hw.constant 1 : i2");
    let _ = writeln!(ir, "  %ph2 = hw.constant 2 : i2");

    // Decode geometry from mux trees (emitted as %mux_* SSA values).
    // %mux_oh, %mux_ow, %mux_oc, %mux_ic, %mux_kh, %mux_kw, %mux_kind, ...
    // Pixel last = oc*oh*ow - 1 → %mux_pix_last
    // Inner last = ic*kh*kw - 1 (or kh*kw for DW, 1 for add/up) → %mux_inner_last

    let _ = writeln!(ir, "  %pix_last_hit = comb.icmp bin eq %pixv, %mux_pix_last : i{pb}");
    let _ = writeln!(ir, "  %inner_last_hit = comb.icmp bin eq %innerv, %mux_inner_last : i{ib}");
    let _ = writeln!(ir, "  %pix_inc = comb.add %pixv, %c1pix : i{pb}");
    let _ = writeln!(ir, "  %inner_inc = comb.add %innerv, %c1in : i{ib}");
    let _ = writeln!(ir, "  %op_inc = comb.add %opv, %c1op : i{opb}");
    let _ = writeln!(ir, "  %op_last_hit = comb.icmp bin eq %opv, %last_op : i{opb}");
    let _ = writeln!(ir, "  %is_ph0 = comb.icmp bin eq %phasev, %ph0 : i2");
    let _ = writeln!(ir, "  %is_ph1 = comb.icmp bin eq %phasev, %ph1 : i2");
    let _ = writeln!(ir, "  %is_ph2 = comb.icmp bin eq %phasev, %ph2 : i2");

    // --- Address / sample / MAC datapath (combinational on current counters) ---
    emit_sample_mac_datapath(ir, ops, &layout, aw, pb, ib, opb, &name)?;

    let _ = writeln!(ir, "  sv.alwaysff(posedge %clk) {{");
    let _ = writeln!(ir, "    sv.if %rst {{");
    let _ = writeln!(ir, "      sv.passign %st, %c0i2 : i2");
    let _ = writeln!(ir, "      sv.passign %done_r, %c0i1 : i1");
    let _ = writeln!(ir, "      sv.passign %busy_r, %c0i1 : i1");
    let _ = writeln!(ir, "      sv.passign %op_r, %c0op : i{opb}");
    let _ = writeln!(ir, "      sv.passign %pix, %c0pix : i{pb}");
    let _ = writeln!(ir, "      sv.passign %inner, %c0in : i{ib}");
    let _ = writeln!(ir, "      sv.passign %acc_r, %c0i32 : i32");
    let _ = writeln!(ir, "      sv.passign %phase, %ph0 : i2");
    let _ = writeln!(ir, "    }} else {{");
    // Host load into input region (only while idle).
    let _ = writeln!(ir, "      sv.if %ld_en {{");
    let _ = writeln!(ir, "        sv.if %is_idle {{");
    let _ = writeln!(ir, "          sv.passign %ld_ptr, %ld_data : i16");
    let _ = writeln!(ir, "        }}");
    let _ = writeln!(ir, "      }}");
    let _ = writeln!(ir, "      sv.if %fire {{");
    let _ = writeln!(ir, "        sv.passign %st, %c1i2 : i2");
    let _ = writeln!(ir, "        sv.passign %busy_r, %c1i1 : i1");
    let _ = writeln!(ir, "        sv.passign %done_r, %c0i1 : i1");
    let _ = writeln!(ir, "        sv.passign %op_r, %c0op : i{opb}");
    let _ = writeln!(ir, "        sv.passign %pix, %c0pix : i{pb}");
    let _ = writeln!(ir, "        sv.passign %inner, %c0in : i{ib}");
    let _ = writeln!(ir, "        sv.passign %phase, %ph0 : i2");
    let _ = writeln!(ir, "        sv.passign %acc_r, %c0i32 : i32");
    let _ = writeln!(ir, "      }} else {{");
    let _ = writeln!(ir, "        sv.if %is_run {{");
    // phase 0: load bias into acc (or 0), go to phase 1 (or 2 for copy-ops)
    let _ = writeln!(ir, "          sv.if %is_ph0 {{");
    let _ = writeln!(ir, "            sv.passign %acc_r, %mux_bias : i32");
    let _ = writeln!(ir, "            sv.passign %inner, %c0in : i{ib}");
    // kind: 0=conv/pw, 1=dw, 2=add, 3=upsample/quant copy
    let _ = writeln!(ir, "            %is_copy = comb.icmp bin uge %mux_kind, %c2i2 : i2");
    let _ = writeln!(ir, "            sv.if %is_copy {{");
    let _ = writeln!(ir, "              sv.passign %phase, %ph2 : i2");
    let _ = writeln!(ir, "            }} else {{");
    let _ = writeln!(ir, "              sv.passign %phase, %ph1 : i2");
    let _ = writeln!(ir, "            }}");
    let _ = writeln!(ir, "          }} else {{");
    let _ = writeln!(ir, "            sv.if %is_ph1 {{");
    // MAC step
    let _ = writeln!(ir, "              sv.passign %acc_r, %mac_next : i32");
    let _ = writeln!(ir, "              sv.if %inner_last_hit {{");
    let _ = writeln!(ir, "                sv.passign %phase, %ph2 : i2");
    let _ = writeln!(ir, "              }} else {{");
    let _ = writeln!(ir, "                sv.passign %inner, %inner_inc : i{ib}");
    let _ = writeln!(ir, "              }}");
    let _ = writeln!(ir, "            }} else {{");
    let _ = writeln!(ir, "              sv.if %is_ph2 {{");
    // Write requant/copy result
    let _ = writeln!(ir, "                %wptr = sv.array_index_inout %mem[%write_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}");
    let _ = writeln!(ir, "                sv.passign %wptr, %write_data : i16");
    let _ = writeln!(ir, "                sv.if %pix_last_hit {{");
    // next op or done
    let _ = writeln!(ir, "                  sv.if %op_last_hit {{");
    let _ = writeln!(ir, "                    sv.passign %st, %c2i2 : i2");
    let _ = writeln!(ir, "                    sv.passign %busy_r, %c0i1 : i1");
    let _ = writeln!(ir, "                    sv.passign %done_r, %c1i1 : i1");
    let _ = writeln!(ir, "                  }} else {{");
    let _ = writeln!(ir, "                    sv.passign %op_r, %op_inc : i{opb}");
    let _ = writeln!(ir, "                    sv.passign %pix, %c0pix : i{pb}");
    let _ = writeln!(ir, "                    sv.passign %phase, %ph0 : i2");
    let _ = writeln!(ir, "                  }}");
    let _ = writeln!(ir, "                }} else {{");
    let _ = writeln!(ir, "                  sv.passign %pix, %pix_inc : i{pb}");
    let _ = writeln!(ir, "                  sv.passign %phase, %ph0 : i2");
    let _ = writeln!(ir, "                }}");
    let _ = writeln!(ir, "              }}");
    let _ = writeln!(ir, "            }}");
    let _ = writeln!(ir, "          }}");
    let _ = writeln!(ir, "        }} else {{");
    // clear done pulse when leaving done state toward idle is automatic via fire path
    let _ = writeln!(ir, "          sv.passign %done_r, %c0i1 : i1");
    let _ = writeln!(ir, "        }}");
    let _ = writeln!(ir, "      }}");
    let _ = writeln!(ir, "    }}");
    let _ = writeln!(ir, "  }}");

    // engines_busy: one-hot of the hybrid engine that owns the current op while busy.
    let opb_busy = bits_for(ops.len().max(1) as u32).max(1);
    let mut eng_bits: Vec<String> = Vec::new();
    for ei in 0..neng {
        let mut acc = format!("eb{ei}_acc");
        let _ = writeln!(ir, "  %{acc} = hw.constant 0 : i1");
        for (oi, &eng) in op_engines.iter().enumerate() {
            if eng != ei {
                continue;
            }
            let eq = format!("eb{ei}_eq{oi}");
            let nxt = format!("eb{ei}_t{oi}");
            let _ = writeln!(ir, "  %c_oi_{ei}_{oi} = hw.constant {oi} : i{opb_busy}");
            let _ = writeln!(
                ir,
                "  %{eq} = comb.icmp bin eq %opv, %c_oi_{ei}_{oi} : i{opb_busy}"
            );
            let _ = writeln!(ir, "  %{nxt} = comb.or %{acc}, %{eq} : i1");
            acc = nxt;
        }
        let _ = writeln!(ir, "  %eb_sel{ei} = comb.and %busyv, %{acc} : i1");
        eng_bits.push(format!("%eb_sel{ei}"));
    }
    let busy_ssa = if neng == 1 {
        eng_bits[0].clone()
    } else {
        let mut cur = eng_bits[0].clone();
        for (i, b) in eng_bits.iter().enumerate().skip(1) {
            let nxt = format!("eb_cat_{i}");
            let _ = writeln!(
                ir,
                "  %{nxt} = comb.concat {b}, {cur} : i1, i{i}"
            );
            cur = format!("%{nxt}");
        }
        cur
    };
    if neng == 1 {
        let _ = writeln!(
            ir,
            "  hw.output %donev, %busyv, {busy_ssa}, %rd_data_v : i1, i1, i1, i16"
        );
    } else {
        let _ = writeln!(
            ir,
            "  hw.output %donev, %busyv, {busy_ssa}, %rd_data_v : i1, i1, i{neng}, i16"
        );
    }
    let _ = writeln!(ir, "}}");
    let _ = writeln!(ir);
    let _ = op_engines; // used above
    Ok(layout)
}

fn cast_zext(ir: &mut String, dst: &str, src: &str, from: u32, to: u32) {
    if from == to {
        let _ = writeln!(ir, "  %c0_{dst} = hw.constant 0 : i{to}");
        let _ = writeln!(ir, "  %{dst} = comb.or %{src}, %c0_{dst} : i{to}");
    } else if from < to {
        let pad = to - from;
        let _ = writeln!(ir, "  %z_{dst} = hw.constant 0 : i{pad}");
        let _ = writeln!(
            ir,
            "  %{dst} = comb.concat %z_{dst}, %{src} : i{pad}, i{from}"
        );
    } else {
        let _ = writeln!(
            ir,
            "  %{dst} = comb.extract %{src} from 0 : (i{from}) -> i{to}"
        );
    }
}

fn emit_op_mux_trees(
    ir: &mut String,
    ops: &[ResolvedOp],
    layout: &MemLayout,
    opb: u32,
    pb: u32,
    ib: u32,
    aw: u32,
) -> Result<(), RtlError> {
    // Build per-op parameter vectors, then a cascade mux on opv.
    let n = ops.len();
    let kinds: Vec<u32> = ops
        .iter()
        .map(|o| match o.kind {
            DopKind::Conv2d | DopKind::PointwiseConv2d => 0,
            DopKind::DepthwiseConv2d => 1,
            DopKind::Add => 2,
            _ => 3, // upsample / quant / relu
        })
        .collect();
    let pix_lasts: Vec<u64> = ops.iter().map(|o| o.out_words().saturating_sub(1)).collect();
    let inner_lasts: Vec<u64> = ops
        .iter()
        .map(|o| {
            let last = match o.kind {
                DopKind::Conv2d | DopKind::PointwiseConv2d => {
                    let icg = (o.ic / o.groups.max(1)).max(1);
                    u64::from(icg) * u64::from(o.kernel[0].max(1)) * u64::from(o.kernel[1].max(1))
                }
                DopKind::DepthwiseConv2d => {
                    u64::from(o.kernel[0].max(1)) * u64::from(o.kernel[1].max(1))
                }
                _ => 1,
            };
            last.saturating_sub(1)
        })
        .collect();
    let out_bases: Vec<u64> = ops
        .iter()
        .map(|o| layout.base_of(Some(o.index)))
        .collect();
    let shifts: Vec<u32> = ops.iter().map(|o| o.shift).collect();
    let clamps_lo: Vec<i64> = ops.iter().map(|o| o.clamp[0]).collect();
    let clamps_hi: Vec<i64> = ops.iter().map(|o| o.clamp[1]).collect();

    mux_u32(ir, "mux_kind", &kinds, "opv", opb, 2);
    mux_u64(ir, "mux_pix_last", &pix_lasts, "opv", opb, pb);
    mux_u64(ir, "mux_inner_last", &inner_lasts, "opv", opb, ib);
    mux_u64(ir, "mux_out_base", &out_bases, "opv", opb, aw);
    // bias is per-channel — handled in sample path; placeholder 0 here for phase0 of copy ops.
    // For conv, bias loaded per channel in sample path via %mux_bias.
    let _ = n;
    let _ = shifts;
    let _ = clamps_lo;
    let _ = clamps_hi;
    Ok(())
}

fn mux_u32(ir: &mut String, name: &str, vals: &[u32], sel: &str, sel_bits: u32, out_bits: u32) {
    if vals.is_empty() {
        let _ = writeln!(ir, "  %{name} = hw.constant 0 : i{out_bits}");
        return;
    }
    for (i, &v) in vals.iter().enumerate() {
        let _ = writeln!(ir, "  %{name}_c{i} = hw.constant {v} : i{out_bits}");
        let _ = writeln!(ir, "  %{name}_i{i} = hw.constant {i} : i{sel_bits}");
        let _ = writeln!(
            ir,
            "  %{name}_eq{i} = comb.icmp bin eq %{sel}, %{name}_i{i} : i{sel_bits}"
        );
    }
    // Cascade: start from last, mux toward 0.
    let last = vals.len() - 1;
    let _ = writeln!(ir, "  %{name}_t{last} = comb.or %{name}_c{last}, %{name}_c{last} : i{out_bits}");
    for i in (0..last).rev() {
        let nxt = i + 1;
        let _ = writeln!(
            ir,
            "  %{name}_t{i} = comb.mux %{name}_eq{i}, %{name}_c{i}, %{name}_t{nxt} : i{out_bits}"
        );
    }
    let _ = writeln!(ir, "  %{name} = comb.or %{name}_t0, %{name}_t0 : i{out_bits}");
}


fn mux_i32_vals(ir: &mut String, name: &str, vals: &[i32], sel: &str, sel_bits: u32, out_bits: u32) {
    if vals.is_empty() {
        let _ = writeln!(ir, "  %{name} = hw.constant 0 : i{out_bits}");
        return;
    }
    for (i, &v) in vals.iter().enumerate() {
        let _ = writeln!(ir, "  %{name}_c{i} = hw.constant {v} : i{out_bits}");
        let _ = writeln!(ir, "  %{name}_i{i} = hw.constant {i} : i{sel_bits}");
        let _ = writeln!(
            ir,
            "  %{name}_eq{i} = comb.icmp bin eq %{sel}, %{name}_i{i} : i{sel_bits}"
        );
    }
    let last = vals.len() - 1;
    let _ = writeln!(ir, "  %{name}_t{last} = comb.or %{name}_c{last}, %{name}_c{last} : i{out_bits}");
    for i in (0..last).rev() {
        let nxt = i + 1;
        let _ = writeln!(
            ir,
            "  %{name}_t{i} = comb.mux %{name}_eq{i}, %{name}_c{i}, %{name}_t{nxt} : i{out_bits}"
        );
    }
    let _ = writeln!(ir, "  %{name} = comb.or %{name}_t0, %{name}_t0 : i{out_bits}");
}

fn mux_u64(ir: &mut String, name: &str, vals: &[u64], sel: &str, sel_bits: u32, out_bits: u32) {
    let v32: Vec<u32> = vals.iter().map(|&v| v as u32).collect();
    mux_u32(ir, name, &v32, sel, sel_bits, out_bits);
}

fn emit_sample_mac_datapath(
    ir: &mut String,
    ops: &[ResolvedOp],
    layout: &MemLayout,
    aw: u32,
    pb: u32,
    ib: u32,
    opb: u32,
    core_name: &str,
) -> Result<(), RtlError> {
    // Decode pix → (channel, oy, ox) and inner → (ic, ky, kx) per op via muxed dims.
    let ohs: Vec<u32> = ops.iter().map(|o| o.oh.max(1)).collect();
    let ows: Vec<u32> = ops.iter().map(|o| o.ow.max(1)).collect();
    let ocs: Vec<u32> = ops.iter().map(|o| o.oc.max(1)).collect();
    let ihs: Vec<u32> = ops.iter().map(|o| o.ih.max(1)).collect();
    let iws: Vec<u32> = ops.iter().map(|o| o.iw.max(1)).collect();
    let ics: Vec<u32> = ops
        .iter()
        .map(|o| match o.kind {
            DopKind::DepthwiseConv2d => 1, // inner has no ic loop; channel from pix
            DopKind::Conv2d | DopKind::PointwiseConv2d => (o.ic / o.groups.max(1)).max(1),
            _ => 1,
        })
        .collect();
    let khs: Vec<u32> = ops.iter().map(|o| o.kernel[0].max(1)).collect();
    let kws: Vec<u32> = ops.iter().map(|o| o.kernel[1].max(1)).collect();
    let pads: Vec<u32> = ops.iter().map(|o| o.pad).collect();
    let s0s: Vec<u32> = ops.iter().map(|o| o.stride[0].max(1)).collect();
    let s1s: Vec<u32> = ops.iter().map(|o| o.stride[1].max(1)).collect();
    let reflects: Vec<u32> = ops.iter().map(|o| u32::from(o.reflect)).collect();
    let in_bases: Vec<u64> = ops
        .iter()
        .map(|o| layout.base_of(o.act_sources.first().copied().flatten()))
        .collect();
    // For Add, second source:
    let in2_bases: Vec<u64> = ops
        .iter()
        .map(|o| layout.base_of(o.act_sources.get(1).copied().flatten()))
        .collect();

    // bits_for(max+1) so the max VALUE itself fits (bits_for(4)==2 only holds 0..3).
    let bw = |v: u32| bits_for(v.saturating_add(1)).max(1);
    mux_u32(ir, "mux_oh", &ohs, "opv", opb, bw(*ohs.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_ow", &ows, "opv", opb, bw(*ows.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_oc", &ocs, "opv", opb, bw(*ocs.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_ih", &ihs, "opv", opb, bw(*ihs.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_iw", &iws, "opv", opb, bw(*iws.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_icg", &ics, "opv", opb, bw(*ics.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_kh", &khs, "opv", opb, bw(*khs.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_kw", &kws, "opv", opb, bw(*kws.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_pad", &pads, "opv", opb, bw(*pads.iter().max().unwrap_or(&1).max(&1)));
    mux_u32(ir, "mux_s0", &s0s, "opv", opb, bw(*s0s.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_s1", &s1s, "opv", opb, bw(*s1s.iter().max().unwrap_or(&1)));
    mux_u32(ir, "mux_reflect", &reflects, "opv", opb, 1);
    mux_u64(ir, "mux_in_base", &in_bases, "opv", opb, aw);
    mux_u64(ir, "mux_in2_base", &in2_bases, "opv", opb, aw);

    let ohb = bw(*ohs.iter().max().unwrap_or(&1));
    let owb = bw(*ows.iter().max().unwrap_or(&1));
    let ocb = bw(*ocs.iter().max().unwrap_or(&1));
    let ihb = bw(*ihs.iter().max().unwrap_or(&1));
    let iwb = bw(*iws.iter().max().unwrap_or(&1));
    let icb = bw(*ics.iter().max().unwrap_or(&1));
    let khb = bw(*khs.iter().max().unwrap_or(&1));
    let kwb = bw(*kws.iter().max().unwrap_or(&1));

    // pix = ((ch * oh) + oy) * ow + ox  → decode with successive rem/div
    // Use widening to i32 for math then extract.
    cast_zext(ir, "pix32", "pixv", pb, 32);
    cast_zext(ir, "oh32", "mux_oh", ohb, 32);
    cast_zext(ir, "ow32", "mux_ow", owb, 32);
    cast_zext(ir, "oc32", "mux_oc", ocb, 32);
    // ox = pix % ow; t1 = pix / ow; oy = t1 % oh; ch = t1 / oh
    let _ = writeln!(ir, "  %ox32 = comb.modu %pix32, %ow32 : i32");
    let _ = writeln!(ir, "  %t1 = comb.divu %pix32, %ow32 : i32");
    let _ = writeln!(ir, "  %oy32 = comb.modu %t1, %oh32 : i32");
    let _ = writeln!(ir, "  %ch32 = comb.divu %t1, %oh32 : i32");

    cast_zext(ir, "inner32", "innerv", ib, 32);
    cast_zext(ir, "kw32", "mux_kw", kwb, 32);
    cast_zext(ir, "kh32", "mux_kh", khb, 32);
    cast_zext(ir, "icg32", "mux_icg", icb, 32);
    // For conv: inner = ((ic * kh) + ky) * kw + kx
    // For dw: inner = ky * kw + kx  (icg muxed to 1)
    let _ = writeln!(ir, "  %kx32 = comb.modu %inner32, %kw32 : i32");
    let _ = writeln!(ir, "  %t2 = comb.divu %inner32, %kw32 : i32");
    let _ = writeln!(ir, "  %ky32 = comb.modu %t2, %kh32 : i32");
    let _ = writeln!(ir, "  %ic32 = comb.divu %t2, %kh32 : i32");

    // Sample coordinate: y = oy*s0 + ky - pad; x = ox*s1 + kx - pad
    cast_zext(ir, "s0_32", "mux_s0", bw(*s0s.iter().max().unwrap_or(&1)), 32);
    cast_zext(ir, "s1_32", "mux_s1", bw(*s1s.iter().max().unwrap_or(&1)), 32);
    cast_zext(ir, "pad32", "mux_pad", bw(*pads.iter().max().unwrap_or(&1).max(&1)), 32);
    cast_zext(ir, "ih32", "mux_ih", ihb, 32);
    cast_zext(ir, "iw32", "mux_iw", iwb, 32);

    let _ = writeln!(ir, "  %y_a = comb.mul %oy32, %s0_32 : i32");
    let _ = writeln!(ir, "  %y_b = comb.add %y_a, %ky32 : i32");
    let _ = writeln!(ir, "  %y_raw = comb.sub %y_b, %pad32 : i32");
    let _ = writeln!(ir, "  %x_a = comb.mul %ox32, %s1_32 : i32");
    let _ = writeln!(ir, "  %x_b = comb.add %x_a, %kx32 : i32");
    let _ = writeln!(ir, "  %x_raw = comb.sub %x_b, %pad32 : i32");

    // Reflect / zero-pad
    emit_reflect_or_oob(ir, "y", "y_raw", "ih32");
    emit_reflect_or_oob(ir, "x", "x_raw", "iw32");
    // mux_reflect selects reflected coords vs raw; samp_valid clears OOB when not reflecting.
    let _ = writeln!(ir, "  %y_idx = comb.mux %mux_reflect, %y_reflected, %y_raw : i32");
    let _ = writeln!(ir, "  %x_idx = comb.mux %mux_reflect, %x_reflected, %x_raw : i32");
    let _ = writeln!(ir, "  %oob = comb.or %y_oob, %x_oob : i1");
    let _ = writeln!(ir, "  %oob_n = comb.xor %oob, %c1i1 : i1");
    let _ = writeln!(ir, "  %samp_valid = comb.or %mux_reflect, %oob_n : i1");

    // Activation channel for sample: DW uses ch; conv uses ic; add/up use ch
    let _ = writeln!(ir, "  %kind_is_dw = comb.icmp bin eq %mux_kind, %c1i2 : i2");
    let _ = writeln!(ir, "  %kind_is_conv = comb.icmp bin eq %mux_kind, %c0i2 : i2");
    let _ = writeln!(ir, "  %samp_ch = comb.mux %kind_is_dw, %ch32, %ic32 : i32");
    // For non-conv kinds in phase2 copy, samp_ch = ch
    let _ = writeln!(ir, "  %samp_ch2 = comb.mux %kind_is_conv, %ic32, %ch32 : i32");
    let _ = writeln!(ir, "  %use_ch = comb.mux %kind_is_dw, %ch32, %samp_ch2 : i32");

    // addr = in_base + (ch * ih + y) * iw + x
    let _ = writeln!(ir, "  %a1 = comb.mul %use_ch, %ih32 : i32");
    let _ = writeln!(ir, "  %a2 = comb.add %a1, %y_idx : i32");
    let _ = writeln!(ir, "  %a3 = comb.mul %a2, %iw32 : i32");
    let _ = writeln!(ir, "  %a4 = comb.add %a3, %x_idx : i32");
    cast_zext(ir, "in_base32", "mux_in_base", aw, 32);
    let _ = writeln!(ir, "  %act_off32 = comb.add %in_base32, %a4 : i32");
    let _ = writeln!(ir, "  %act_abs = comb.extract %act_off32 from 0 : (i32) -> i{aw}");
    let words = layout.total_words;
    let _ = writeln!(
        ir,
        "  %act_ptr = sv.array_index_inout %mem[%act_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}"
    );
    let _ = writeln!(ir, "  %act_i16 = sv.read_inout %act_ptr : !hw.inout<i16>");
    // Zero if OOB and not reflect (keep i16 — unclamped add can exceed i8).
    let _ = writeln!(ir, "  %c0i16_act = hw.constant 0 : i16");
    let _ = writeln!(ir, "  %act_eff = comb.mux %samp_valid, %act_i16, %c0i16_act : i16");

    // Weight address: conv (ch*icg+ic)*kh*kw + ky*kw + kx; dw (ch*kh+ky)*kw + kx
    let _ = writeln!(ir, "  %w_a = comb.mul %ch32, %icg32 : i32");
    let _ = writeln!(ir, "  %w_b = comb.add %w_a, %ic32 : i32");
    let _ = writeln!(ir, "  %w_ch = comb.mux %kind_is_dw, %ch32, %w_b : i32");
    let _ = writeln!(ir, "  %w_c = comb.mul %w_ch, %kh32 : i32");
    let _ = writeln!(ir, "  %w_d = comb.add %w_c, %ky32 : i32");
    let _ = writeln!(ir, "  %w_e = comb.mul %w_d, %kw32 : i32");
    let _ = writeln!(ir, "  %w_idx32 = comb.add %w_e, %kx32 : i32");

    // Per-op weight ROM instance mux: instantiate all, mux data by opv.
    let max_w = ops.iter().map(|o| o.weights.len().max(1)).max().unwrap_or(1);
    let waw = bits_for(max_w as u32).max(1);
    let _ = writeln!(ir, "  %w_idx = comb.extract %w_idx32 from 0 : (i32) -> i{waw}");
    for op in ops {
        if op.weights.is_empty() {
            let _ = writeln!(ir, "  %wdat_{0} = hw.constant 0 : i8", op.index);
        } else {
            let rom = format!("{core_name}_w{0}", op.index);
            let naw = bits_for(op.weights.len().max(1) as u32).max(1);
            if naw == waw {
                let _ = writeln!(
                    ir,
                    "  %wdat_{0} = hw.instance \"u_w{0}\" @{rom}(addr: %w_idx: i{waw}) -> (data: i8)",
                    op.index
                );
            } else {
                cast_zext(ir, &format!("widx_{0}", op.index), "w_idx", waw, naw);
                let _ = writeln!(
                    ir,
                    "  %wdat_{0} = hw.instance \"u_w{0}\" @{rom}(addr: %widx_{0}: i{naw}) -> (data: i8)",
                    op.index
                );
            }
        }
    }
    // Mux weight data
    mux_i8_by_op(ir, "w_data", ops, opb);

    // Bias ROM mux by channel
    let max_b = ops.iter().map(|o| o.bias.len().max(1)).max().unwrap_or(1);
    let baw = bits_for(max_b as u32).max(1);
    let _ = writeln!(ir, "  %ch_b = comb.extract %ch32 from 0 : (i32) -> i{baw}");
    for op in ops {
        if op.bias.is_empty() {
            let _ = writeln!(ir, "  %bdat_{0} = hw.constant 0 : i32", op.index);
        } else {
            let rom = format!("{core_name}_b{0}", op.index);
            let naw = bits_for(op.bias.len().max(1) as u32).max(1);
            if naw == baw {
                let _ = writeln!(
                    ir,
                    "  %bdat_{0} = hw.instance \"u_b{0}\" @{rom}(addr: %ch_b: i{baw}) -> (data: i32)",
                    op.index
                );
            } else {
                cast_zext(ir, &format!("bidx_{0}", op.index), "ch_b", baw, naw);
                let _ = writeln!(
                    ir,
                    "  %bdat_{0} = hw.instance \"u_b{0}\" @{rom}(addr: %bidx_{0}: i{naw}) -> (data: i32)",
                    op.index
                );
            }
        }
    }
    mux_i32_by_op(ir, "mux_bias", ops, "bdat", opb);

    // M0 ROM
    let max_m = ops.iter().map(|o| o.m0.len().max(1)).max().unwrap_or(1);
    let maw = bits_for(max_m as u32).max(1);
    let _ = writeln!(ir, "  %ch_m = comb.extract %ch32 from 0 : (i32) -> i{maw}");
    for op in ops {
        if op.m0.is_empty() {
            let _ = writeln!(ir, "  %mdat_{0} = hw.constant 1 : i32", op.index);
        } else {
            let rom = format!("{core_name}_m{0}", op.index);
            let naw = bits_for(op.m0.len().max(1) as u32).max(1);
            if naw == maw {
                let _ = writeln!(
                    ir,
                    "  %mdat_{0} = hw.instance \"u_m{0}\" @{rom}(addr: %ch_m: i{maw}) -> (data: i32)",
                    op.index
                );
            } else {
                cast_zext(ir, &format!("midx_{0}", op.index), "ch_m", maw, naw);
                let _ = writeln!(
                    ir,
                    "  %mdat_{0} = hw.instance \"u_m{0}\" @{rom}(addr: %midx_{0}: i{naw}) -> (data: i32)",
                    op.index
                );
            }
        }
    }
    mux_i32_by_op(ir, "mux_m0", ops, "mdat", opb);

    // MAC: acc + sext(act)*sext(w)
    let _ = writeln!(
        ir,
        "  %mac_next = hw.instance \"u_mac\" @tessera_mac_lane(a: %act_eff: i16, b: %w_data: i8, acc: %accv: i32) -> (o: i32)"
    );

    // Write data: requant for conv/dw/add-branch; copy/upsample for others; add = trunk+scaled
    // Clamp constants mux
    let lo: Vec<i32> = ops
        .iter()
        .map(|o| o.clamp[0].clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32)
        .collect();
    let hi: Vec<i32> = ops
        .iter()
        .map(|o| o.clamp[1].clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32)
        .collect();
    mux_i32_vals(ir, "mux_lo", &lo, "opv", opb, 16);
    mux_i32_vals(ir, "mux_hi", &hi, "opv", opb, 16);

    // Per-op shift requant instances, mux result.
    for op in ops {
        let rq = format!("tessera_requant_s{}", op.shift);
        // For add: requant the branch sample (act from in2), not acc.
        let _ = writeln!(
            ir,
            "  %rq_{0} = hw.instance \"u_rq{0}\" @{rq}(acc: %accv: i32, m0: %mux_m0: i32, lo: %mux_lo: i16, hi: %mux_hi: i16) -> (o: i16)",
            op.index
        );
    }
    mux_i16_by_op(ir, "rq_data", ops, "rq", opb);

    // Add path: trunk + requant(branch). Branch loaded from in2; trunk from in_base.
    // For phase2 of add: we treat write_data specially.
    // Load trunk at pix address from in_base (ch,oy,ox) — no pad
    let _ = writeln!(ir, "  %tr_a = comb.mul %ch32, %ih32 : i32");
    let _ = writeln!(ir, "  %tr_b = comb.add %tr_a, %oy32 : i32");
    let _ = writeln!(ir, "  %tr_c = comb.mul %tr_b, %iw32 : i32");
    let _ = writeln!(ir, "  %tr_d = comb.add %tr_c, %ox32 : i32");
    cast_zext(ir, "in2_base32", "mux_in2_base", aw, 32);
    let _ = writeln!(ir, "  %trunk_off = comb.add %in_base32, %tr_d : i32");
    let _ = writeln!(ir, "  %branch_off = comb.add %in2_base32, %tr_d : i32");
    let _ = writeln!(ir, "  %trunk_abs = comb.extract %trunk_off from 0 : (i32) -> i{aw}");
    let _ = writeln!(ir, "  %branch_abs = comb.extract %branch_off from 0 : (i32) -> i{aw}");
    let _ = writeln!(
        ir,
        "  %trunk_ptr = sv.array_index_inout %mem[%trunk_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}"
    );
    let _ = writeln!(
        ir,
        "  %branch_ptr = sv.array_index_inout %mem[%branch_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}"
    );
    let _ = writeln!(ir, "  %trunk_v = sv.read_inout %trunk_ptr : !hw.inout<i16>");
    let _ = writeln!(ir, "  %branch_v = sv.read_inout %branch_ptr : !hw.inout<i16>");
    // Requant branch as i32 acc
    let _ = writeln!(ir, "  %br_s = comb.extract %branch_v from 15 : (i16) -> i1");
    let _ = writeln!(ir, "  %br_r = comb.replicate %br_s : (i1) -> i16");
    let _ = writeln!(ir, "  %br32 = comb.concat %br_r, %branch_v : i16, i16");
    for op in ops {
        let rq = format!("tessera_requant_s{}", op.shift);
        let _ = writeln!(
            ir,
            "  %rqbr_{0} = hw.instance \"u_rqbr{0}\" @{rq}(acc: %br32: i32, m0: %mux_m0: i32, lo: %mux_lo: i16, hi: %mux_hi: i16) -> (o: i16)",
            op.index
        );
    }
    mux_i16_by_op(ir, "rq_branch", ops, "rqbr", opb);
    let _ = writeln!(ir, "  %add_sum = comb.add %trunk_v, %rq_branch : i16");

    // Upsample: read from (ch, oy/2, ox/2)
    let _ = writeln!(ir, "  %oy_h = comb.shru %oy32, %c1i32_ref : i32");
    let _ = writeln!(ir, "  %ox_h = comb.shru %ox32, %c1i32_ref : i32");
    // ih/iw for upsample source are mux_ih/iw already
    let _ = writeln!(ir, "  %up_a = comb.mul %ch32, %ih32 : i32");
    let _ = writeln!(ir, "  %up_b = comb.add %up_a, %oy_h : i32");
    let _ = writeln!(ir, "  %up_c = comb.mul %up_b, %iw32 : i32");
    let _ = writeln!(ir, "  %up_d = comb.add %up_c, %ox_h : i32");
    let _ = writeln!(ir, "  %up_off = comb.add %in_base32, %up_d : i32");
    let _ = writeln!(ir, "  %up_abs = comb.extract %up_off from 0 : (i32) -> i{aw}");
    let _ = writeln!(
        ir,
        "  %up_ptr = sv.array_index_inout %mem[%up_abs] : !hw.inout<uarray<{words}xi16>>, i{aw}"
    );
    let _ = writeln!(ir, "  %up_v = sv.read_inout %up_ptr : !hw.inout<i16>");

    // kind select write_data
    let _ = writeln!(ir, "  %kind_is_add = comb.icmp bin eq %mux_kind, %c2i2 : i2");
    let _ = writeln!(ir, "  %wd_a = comb.mux %kind_is_add, %add_sum, %rq_data : i16");
    let _ = writeln!(ir, "  %kind_is_copy = comb.icmp bin eq %mux_kind, %c3i2 : i2");
    // Upsample when oh != ih; else quantize/relu copy. Clamp copy into [lo,hi].
    let _ = writeln!(ir, "  %oh_ne_ih = comb.icmp bin ne %oh32, %ih32 : i32");
    let _ = writeln!(ir, "  %copy_raw = comb.mux %oh_ne_ih, %up_v, %trunk_v : i16");
    // Quantize/relu clamp; upsample is a pure nearest copy (no clamp in reference).
    let _ = writeln!(ir, "  %copy_lt = comb.icmp bin slt %copy_raw, %mux_lo : i16");
    let _ = writeln!(ir, "  %copy_gt = comb.icmp bin sgt %copy_raw, %mux_hi : i16");
    let _ = writeln!(ir, "  %copy_lo = comb.mux %copy_lt, %mux_lo, %copy_raw : i16");
    let _ = writeln!(ir, "  %copy_clamped = comb.mux %copy_gt, %mux_hi, %copy_lo : i16");
    let _ = writeln!(ir, "  %copy_v = comb.mux %oh_ne_ih, %copy_raw, %copy_clamped : i16");
    let _ = writeln!(ir, "  %write_data = comb.mux %kind_is_copy, %copy_v, %wd_a : i16");

    // write_abs = out_base + pix
    cast_zext(ir, "pix_aw", "pixv", pb, aw);
    let _ = writeln!(ir, "  %write_abs = comb.add %mux_out_base, %pix_aw : i{aw}");

    let _ = ib; // used
    Ok(())
}

fn emit_reflect_or_oob(ir: &mut String, prefix: &str, raw: &str, extent: &str) {
    if !ir.contains("%c1i32_ref =") {
        let _ = writeln!(ir, "  %c1i32_ref = hw.constant 1 : i32");
    }
    let _ = writeln!(ir, "  %c0_{prefix} = hw.constant 0 : i32");
    let _ = writeln!(ir, "  %{prefix}_neg = comb.icmp bin slt %{raw}, %c0_{prefix} : i32");
    let _ = writeln!(ir, "  %{prefix}_abs = comb.sub %c0_{prefix}, %{raw} : i32");
    let _ = writeln!(ir, "  %{prefix}_c2 = hw.constant 2 : i32");
    let _ = writeln!(ir, "  %{prefix}_e1 = comb.sub %{extent}, %c1i32_ref : i32");
    let _ = writeln!(ir, "  %{prefix}_span = comb.mul %{prefix}_c2, %{prefix}_e1 : i32");
    let _ = writeln!(ir, "  %{prefix}_mir = comb.sub %{prefix}_span, %{raw} : i32");
    let _ = writeln!(ir, "  %{prefix}_ge = comb.icmp bin sge %{raw}, %{extent} : i32");
    let _ = writeln!(
        ir,
        "  %{prefix}_fold = comb.mux %{prefix}_neg, %{prefix}_abs, %{raw} : i32"
    );
    // When reflecting: use folded abs / mirror; else keep raw and mark OOB.
    let _ = writeln!(
        ir,
        "  %{prefix}_reflected = comb.mux %{prefix}_ge, %{prefix}_mir, %{prefix}_fold : i32"
    );
    let _ = writeln!(
        ir,
        "  %{prefix}_oob = comb.or %{prefix}_neg, %{prefix}_ge : i1"
    );
}

fn mux_i8_by_op(ir: &mut String, name: &str, ops: &[ResolvedOp], opb: u32) {
    let n = ops.len();
    if n == 0 {
        let _ = writeln!(ir, "  %{name} = hw.constant 0 : i8");
        return;
    }
    for (i, op) in ops.iter().enumerate() {
        let _ = writeln!(ir, "  %{name}_i{i} = hw.constant {i} : i{opb}");
        let _ = writeln!(
            ir,
            "  %{name}_eq{i} = comb.icmp bin eq %opv, %{name}_i{i} : i{opb}"
        );
        let _ = (op,);
    }
    let last = n - 1;
    let _ = writeln!(
        ir,
        "  %{name}_t{last} = comb.or %wdat_{}, %wdat_{} : i8",
        ops[last].index,
        ops[last].index
    );
    for i in (0..last).rev() {
        let nxt = i + 1;
        let _ = writeln!(
            ir,
            "  %{name}_t{i} = comb.mux %{name}_eq{i}, %wdat_{}, %{name}_t{nxt} : i8",
            ops[i].index
        );
    }
    let _ = writeln!(ir, "  %{name} = comb.or %{name}_t0, %{name}_t0 : i8");
}

fn mux_i32_by_op(ir: &mut String, name: &str, ops: &[ResolvedOp], prefix: &str, opb: u32) {
    let n = ops.len();
    if n == 0 {
        let _ = writeln!(ir, "  %{name} = hw.constant 0 : i32");
        return;
    }
    for (i, _) in ops.iter().enumerate() {
        let _ = writeln!(ir, "  %{name}_i{i} = hw.constant {i} : i{opb}");
        let _ = writeln!(
            ir,
            "  %{name}_eq{i} = comb.icmp bin eq %opv, %{name}_i{i} : i{opb}"
        );
    }
    let last = n - 1;
    let _ = writeln!(
        ir,
        "  %{name}_t{last} = comb.or %{prefix}_{}, %{prefix}_{} : i32",
        ops[last].index,
        ops[last].index
    );
    for i in (0..last).rev() {
        let nxt = i + 1;
        let _ = writeln!(
            ir,
            "  %{name}_t{i} = comb.mux %{name}_eq{i}, %{prefix}_{}, %{name}_t{nxt} : i32",
            ops[i].index
        );
    }
    let _ = writeln!(ir, "  %{name} = comb.or %{name}_t0, %{name}_t0 : i32");
}

fn mux_i16_by_op(ir: &mut String, name: &str, ops: &[ResolvedOp], prefix: &str, opb: u32) {
    let n = ops.len();
    if n == 0 {
        let _ = writeln!(ir, "  %{name} = hw.constant 0 : i16");
        return;
    }
    for (i, _) in ops.iter().enumerate() {
        let _ = writeln!(ir, "  %{name}_i{i} = hw.constant {i} : i{opb}");
        let _ = writeln!(
            ir,
            "  %{name}_eq{i} = comb.icmp bin eq %opv, %{name}_i{i} : i{opb}"
        );
    }
    let last = n - 1;
    let _ = writeln!(
        ir,
        "  %{name}_t{last} = comb.or %{prefix}_{}, %{prefix}_{} : i16",
        ops[last].index,
        ops[last].index
    );
    for i in (0..last).rev() {
        let nxt = i + 1;
        let _ = writeln!(
            ir,
            "  %{name}_t{i} = comb.mux %{name}_eq{i}, %{prefix}_{}, %{name}_t{nxt} : i16",
            ops[i].index
        );
    }
    let _ = writeln!(ir, "  %{name} = comb.or %{name}_t0, %{name}_t0 : i16");
}

/// Wire engines + arenas into the top; OR real busy; optional sched_core for
/// bit-exact rd_data when `has_core`.
/// Top DUT: sole datapath is `*_sched_core` (no stub engines).
///
/// Hybrid engine indices still appear on `engines_busy`, decoded from the
/// schedule core's current op via the solved binds — structural CP-SAT
/// decisions remain visible without a second compute path.
pub fn emit_sched_top(
    ir: &mut String,
    n_engines: usize,
    in_aw: u32,
    out_aw: u32,
    dut_name: &str,
) {
    let neng = n_engines.max(1);
    let busy_ty = if neng == 1 {
        "i1".into()
    } else {
        format!("i{neng}")
    };
    let core = format!("{dut_name}_sched_core");
    let _ = writeln!(
        ir,
        "// top — single datapath: integer schedule core (hybrid eng_busy from binds)"
    );
    let _ = writeln!(
        ir,
        "hw.module @{dut_name}(in %clk : i1, in %rst : i1, in %start : i1, out done : i1, out engines_busy : {busy_ty}, in %ld_en : i1, in %ld_addr : i{in_aw}, in %ld_data : i16, in %rd_addr : i{out_aw}, out rd_data : i16) {{"
    );
    let _ = writeln!(
        ir,
        "  %done_o, %busy_o, %eng_busy, %rdata = hw.instance \"u_sched\" @{core}(clk: %clk: i1, rst: %rst: i1, start: %start: i1, ld_en: %ld_en: i1, ld_addr: %ld_addr: i{in_aw}, ld_data: %ld_data: i16, rd_addr: %rd_addr: i{out_aw}) -> (done: i1, busy: i1, engines_busy: {busy_ty}, rd_data: i16)"
    );
    let _ = writeln!(ir, "  %_busy_touch = comb.xor %busy_o, %busy_o : i1");
    if neng == 1 {
        let _ = writeln!(
            ir,
            "  hw.output %done_o, %eng_busy, %rdata : i1, i1, i16"
        );
    } else {
        let _ = writeln!(
            ir,
            "  hw.output %done_o, %eng_busy, %rdata : i1, i{neng}, i16"
        );
    }
    let _ = writeln!(ir, "}}");
}

