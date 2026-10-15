//! CIRCT dialects inside the Melior context, and in-process ExportVerilog.

use std::ffi::c_void;

use super::RtlError;

/// Register and load `hw`, `comb`, `seq`, `sv`, `esi` into `context`.
pub fn load_circt_dialects(context: &melior::Context) {
    let raw = context.to_raw();
    unsafe {
        for handle in [
            circt_sys::mlirGetDialectHandle__hw__(),
            circt_sys::mlirGetDialectHandle__comb__(),
            circt_sys::mlirGetDialectHandle__seq__(),
            circt_sys::mlirGetDialectHandle__sv__(),
            circt_sys::mlirGetDialectHandle__esi__(),
        ] {
            mlir_sys::mlirDialectHandleRegisterDialect(handle, raw);
            mlir_sys::mlirDialectHandleLoadDialect(handle, raw);
        }
    }
}

unsafe extern "C" fn sink(s: mlir_sys::MlirStringRef, data: *mut c_void) {
    let out = unsafe { &mut *(data as *mut String) };
    let bytes = unsafe { std::slice::from_raw_parts(s.data as *const u8, s.length) };
    out.push_str(&String::from_utf8_lossy(bytes));
}

/// Run CIRCT ExportVerilog on `module` in-process and return the SystemVerilog.
pub fn export_verilog(module: &melior::ir::Module) -> Result<String, RtlError> {
    let mut out = String::new();
    let result = unsafe {
        circt_sys::mlirExportVerilog(
            module.to_raw(),
            Some(sink),
            &mut out as *mut String as *mut c_void,
        )
    };
    if result.value == 0 {
        return Err(RtlError::UnsupportedLayout {
            detail: "CIRCT ExportVerilog failed".to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::{Location, Module, block::BlockLike, operation::{OperationBuilder, OperationLike}, RegionLike};

    /// Build `hw.module` + `comb.add` with Melior's OperationBuilder (no text IR)
    /// and export it in-process.
    #[test]
    fn melior_built_hw_module_exports() {
        let context = melior::Context::new();
        load_circt_dialects(&context);
        let module = Module::parse(
            &context,
            "hw.module @add(in %a : i8, in %b : i8, out o : i8) {\n  hw.output %a : i8\n}\n",
        )
        .expect("parse skeleton");
        // Rewrite the body through the builder: o = a + b.
        let body_module = module.body().first_operation().expect("hw.module");
        let region = body_module.region(0).expect("body");
        let block = region.first_block().expect("entry");
        let a = block.argument(0).unwrap().into();
        let b = block.argument(1).unwrap().into();
        let loc = Location::unknown(&context);
        let terminator = block.terminator().expect("hw.output");
        let add = OperationBuilder::new("comb.add", loc)
            .add_operands(&[a, b])
            .add_results(&[melior::ir::r#type::IntegerType::new(&context, 8).into()])
            .build()
            .expect("comb.add");
        let add = block.insert_operation_before(terminator, add);
        let sum = add.result(0).unwrap().into();
        let out = OperationBuilder::new("hw.output", loc)
            .add_operands(&[sum])
            .build()
            .expect("hw.output");
        block.insert_operation_before(terminator, out);
        unsafe { mlir_sys::mlirOperationDestroy(terminator.to_raw()) };
        assert!(module.as_operation().verify(), "verify");
        let sv = export_verilog(&module).expect("export");
        assert!(sv.contains("module add"), "{sv}");
        assert!(sv.contains("a + b"), "{sv}");
    }
}
