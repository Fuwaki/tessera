//! Minimal hand-written bindings to the CIRCT C API.
//!
//! Only what Tessera's tessera→CIRCT lowering needs: dialect handles for
//! `hw`, `comb`, `seq`, `sv`, `esi`, the pass registrations used before
//! export, a few type constructors, and `mlirExportVerilog`.

#![allow(non_snake_case)]

pub use mlir_sys::{
    MlirContext, MlirDialectHandle, MlirLogicalResult, MlirModule, MlirStringCallback, MlirType,
};

unsafe extern "C" {
    pub fn mlirGetDialectHandle__hw__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__comb__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__seq__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__sv__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__esi__() -> MlirDialectHandle;

    pub fn registerSeqPasses();
    pub fn registerSVPasses();
    pub fn registerHWPasses();
    pub fn mlirRegisterCIRCTConversionPasses();

    pub fn seqClockTypeGet(ctx: MlirContext) -> MlirType;
    pub fn hwArrayTypeGet(element: MlirType, size: usize) -> MlirType;
    pub fn hwInOutTypeGet(element: MlirType) -> MlirType;

    pub fn mlirExportVerilog(
        module: MlirModule,
        callback: MlirStringCallback,
        user_data: *mut std::ffi::c_void,
    ) -> MlirLogicalResult;
}
