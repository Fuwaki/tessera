//! Register CIRCT dialects into a bare MLIR context, parse hw/comb, export SV.
use circt_sys::*;
use mlir_sys::*;
use std::ffi::c_void;

unsafe extern "C" fn sink(s: MlirStringRef, data: *mut c_void) {
    let out = unsafe { &mut *(data as *mut String) };
    let bytes = unsafe { std::slice::from_raw_parts(s.data as *const u8, s.length) };
    out.push_str(&String::from_utf8_lossy(bytes));
}

#[test]
fn export_verilog_in_process() {
    unsafe {
        let ctx = mlirContextCreate();
        for h in [mlirGetDialectHandle__hw__(), mlirGetDialectHandle__comb__(),
                  mlirGetDialectHandle__seq__(), mlirGetDialectHandle__sv__()] {
            mlirDialectHandleRegisterDialect(h, ctx);
            mlirDialectHandleLoadDialect(h, ctx);
        }
        let src = "hw.module @add(in %a : i8, in %b : i8, out o : i8) {\n %0 = comb.add %a, %b : i8\n hw.output %0 : i8\n}\n";
        let m = mlirModuleCreateParse(ctx, MlirStringRef { data: src.as_ptr() as _, length: src.len() });
        assert!(!m.ptr.is_null(), "parse");
        let mut out = String::new();
        let r = mlirExportVerilog(m, Some(sink), &mut out as *mut String as *mut c_void);
        assert!(r.value != 0, "export failed");
        assert!(out.contains("module add"), "{out}");
        mlirModuleDestroy(m);
        mlirContextDestroy(ctx);
    }
}
