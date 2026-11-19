// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-independent FMI 2/3 Co-Simulation execution (private C ABI).
use fmi_rs::fmi2::{
    CS, FMU2,
    builder::FMU2Builder,
    types::{fmi2Status, fmi2True},
};
use fmi_rs::fmi3::{FMU3, builder::FMU3Builder, types::fmi3Status};
use fmi_rs::model_description::{self, fmi2 as md2, fmi3 as md3};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

// Drop order: free the instance before releasing its extracted resources.
enum Model {
    Fmi2 {
        fmu: Arc<FMU2<CS>>,
        builder: FMU2Builder,
    },
    Fmi3 {
        fmu: Arc<FMU3>,
        builder: FMU3Builder,
    },
}
struct Host {
    model: Model,
    time: f64,
    initialized: bool,
    failed: bool,
    pending_inputs: BTreeMap<u32, f64>,
}

fn ok2(status: fmi2Status, operation: &str) -> Result<(), String> {
    match status {
        fmi2Status::Ok | fmi2Status::Warning => Ok(()),
        other => Err(format!("{operation} failed with {other:?}")),
    }
}
fn ok3(status: fmi3Status, operation: &str) -> Result<(), String> {
    match status {
        fmi3Status::Ok | fmi3Status::Warning => Ok(()),
        other => Err(format!("{operation} failed with {other:?}")),
    }
}

// Own unpacked FMUs just like extracted archives; never follow directory links.
fn copy_directory(source: &Path, target: &Path) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(std::io::Error::other(
                "links in unpacked FMUs are unsupported",
            ));
        }
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            copy_directory(&entry.path(), &destination)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), destination)?;
        }
    }
    Ok(())
}

impl Host {
    fn new(path: &Path) -> Result<Self, String> {
        let unzipdir = tempfile::tempdir().map_err(|e| e.to_string())?;
        if path.is_dir() {
            copy_directory(path, unzipdir.path()).map_err(|e| e.to_string())?;
        } else {
            fmi_rs::zip::extract_zip_archive(path, &unzipdir).map_err(|e| e.to_string())?;
        }
        let xml = unzipdir.path().join("modelDescription.xml");
        let version = model_description::peek_fmi_version(&xml).map_err(|e| e.to_string())?;
        let model = if version == "2.0" {
            let builder = FMU2Builder {
                model_description: md2::ModelDescription::from_path(&xml)
                    .map_err(|e| e.to_string())?,
                unzipdir,
                visible: false,
                loggingOn: false,
                logFile: None,
                logCalls: false,
            };
            let fmu = builder.instantiate_cs("ovfmi").map_err(|e| e.to_string())?;
            Model::Fmi2 { fmu, builder }
        } else if version.starts_with("3.0") {
            let builder = FMU3Builder {
                model_description: md3::ModelDescription::from_path(&xml)
                    .map_err(|e| e.to_string())?,
                unzipdir,
                visible: false,
                loggingOn: false,
                logFile: None,
                logCalls: false,
                printMessages: true,
                // The CS FMU handles events internally. This is not an ME solver.
                eventModeUsed: false,
                earlyReturnAllowed: false,
                requiredIntermediateVariables: vec![],
            };
            let fmu = builder.instantiate_cs("ovfmi").map_err(|e| e.to_string())?;
            Model::Fmi3 { fmu, builder }
        } else {
            return Err(format!(
                "unsupported FMI version {version}; expected FMI 2/3 Co-Simulation"
            ));
        };
        Ok(Self {
            model,
            time: 0.0,
            initialized: false,
            failed: false,
            pending_inputs: BTreeMap::new(),
        })
    }

    fn initialize(&mut self) -> Result<(), String> {
        if !self.initialized {
            match &self.model {
                Model::Fmi2 { fmu, .. } => {
                    ok2(fmu.setupExperiment(None, 0.0, None), "fmi2SetupExperiment")?;
                    ok2(fmu.enterInitializationMode(), "fmi2EnterInitializationMode")?;
                    for (&reference, &value) in &self.pending_inputs {
                        ok2(fmu.setReal(&[reference], &[value]), "fmi2SetReal")?;
                    }
                    ok2(fmu.exitInitializationMode(), "fmi2ExitInitializationMode")?;
                }
                Model::Fmi3 { fmu, .. } => {
                    ok3(
                        fmu.enterInitializationMode(None, 0.0, None),
                        "fmi3EnterInitializationMode",
                    )?;
                    ok3(fmu.exitInitializationMode(), "fmi3ExitInitializationMode")?;
                }
            }
            self.pending_inputs.clear();
            self.initialized = true;
        }
        Ok(())
    }

    fn set_real(&mut self, name: &str, value: f64) -> Result<(), String> {
        if self.failed {
            return Err("FMU failed; reset or recreate it before continuing".into());
        }
        match &self.model {
            Model::Fmi2 { fmu, builder } => {
                let v = builder
                    .model_description
                    .variable_by_name(name)
                    .map_err(|_| format!("unknown FMI variable {name:?}"))?;
                if !matches!(v.variableType, md2::VariableType::Real { .. }) {
                    return Err(format!("{name}: native routing requires FMI 2 Real"));
                }
                // FMI 2 inputs can be set only after initialization starts.
                if !self.initialized && v.causality == md2::Causality::Input {
                    self.pending_inputs.insert(v.valueReference, value);
                    return Ok(());
                }
                if self.initialized
                    && (v.variability == md2::Variability::Fixed
                        || v.causality == md2::Causality::Output)
                {
                    return Ok(());
                }
                ok2(fmu.setReal(&[v.valueReference], &[value]), "fmi2SetReal")
            }
            Model::Fmi3 { fmu, builder } => {
                let v = builder
                    .model_description
                    .variable_by_name(name)
                    .map_err(|_| format!("unknown FMI variable {name:?}"))?;
                if !matches!(v.variableType, md3::VariableType::Float64 { .. })
                    || !v.dimensions.is_empty()
                {
                    return Err(format!(
                        "{name}: native routing requires scalar FMI 3 Float64"
                    ));
                }
                if self.initialized
                    && (v.variability == md3::Variability::Fixed
                        || v.causality == md3::Causality::Output)
                {
                    return Ok(());
                }
                ok3(
                    fmu.setFloat64(&[v.valueReference], &[value]),
                    "fmi3SetFloat64",
                )
            }
        }
    }

    fn get_real(&self, name: &str) -> Result<f64, String> {
        let mut result = [0.0];
        match &self.model {
            Model::Fmi2 { fmu, builder } => {
                let v = builder
                    .model_description
                    .variable_by_name(name)
                    .map_err(|_| format!("unknown FMI variable {name:?}"))?;
                if !matches!(v.variableType, md2::VariableType::Real { .. }) {
                    return Err(format!("{name}: native routing requires FMI 2 Real"));
                }
                ok2(fmu.getReal(&[v.valueReference], &mut result), "fmi2GetReal")?;
            }
            Model::Fmi3 { fmu, builder } => {
                let v = builder
                    .model_description
                    .variable_by_name(name)
                    .map_err(|_| format!("unknown FMI variable {name:?}"))?;
                if !matches!(v.variableType, md3::VariableType::Float64 { .. })
                    || !v.dimensions.is_empty()
                {
                    return Err(format!(
                        "{name}: native routing requires scalar FMI 3 Float64"
                    ));
                }
                ok3(
                    fmu.getFloat64(&[v.valueReference], &mut result),
                    "fmi3GetFloat64",
                )?;
            }
        }
        Ok(result[0])
    }

    fn step(&mut self, target: f64) -> Result<(), String> {
        if !target.is_finite() || target <= self.time {
            return Err(format!(
                "target time {target} must be greater than {}",
                self.time
            ));
        }
        if self.failed {
            return Err("FMU failed; reset or recreate it before continuing".into());
        }
        // Failed FMI calls can partially advance internal state; disallow reuse.
        self.failed = true;
        self.initialize()?;
        let dt = target - self.time;
        match &self.model {
            Model::Fmi2 { fmu, .. } => ok2(fmu.doStep(self.time, dt, fmi2True), "fmi2DoStep")?,
            Model::Fmi3 { fmu, .. } => {
                let (mut event, mut terminate, mut early, mut last) =
                    (false, false, false, self.time);
                ok3(
                    fmu.doStep(
                        self.time,
                        dt,
                        true,
                        &mut event,
                        &mut terminate,
                        &mut early,
                        &mut last,
                    ),
                    "fmi3DoStep",
                )?;
                if event || early || terminate {
                    return Err(format!(
                        "FMU requested unsupported external event/early-return/termination (event={event}, early={early}, terminate={terminate}, last={last})"
                    ));
                }
            }
        }
        self.time = target;
        self.failed = false;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), String> {
        match &self.model {
            Model::Fmi2 { fmu, .. } => ok2(fmu.reset(), "fmi2Reset")?,
            Model::Fmi3 { fmu, .. } => ok3(fmu.reset(), "fmi3Reset")?,
        }
        self.time = 0.0;
        self.initialized = false;
        self.failed = false;
        self.pending_inputs.clear();
        Ok(())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        if self.initialized && !self.failed {
            match &self.model {
                Model::Fmi2 { fmu, .. } => {
                    let _ = fmu.terminate();
                }
                Model::Fmi3 { fmu, .. } => {
                    let _ = fmu.terminate();
                }
            }
        }
    }
}

fn set_error(message: impl ToString) {
    let cleaned = message.to_string().replace('\0', " ");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(cleaned).unwrap());
}
unsafe fn string_arg<'a>(value: *const c_char, name: &str) -> Result<&'a str, String> {
    if value.is_null() {
        return Err(format!("{name} must not be null"));
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|error| format!("{name} is not UTF-8: {error}"))
}
unsafe fn host_mut<'a>(handle: *mut c_void) -> Result<&'a mut Host, String> {
    if handle.is_null() {
        Err("host handle must not be null".to_string())
    } else {
        Ok(unsafe { &mut *(handle as *mut Host) })
    }
}
fn guarded<T>(default: T, operation: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_error(error);
            default
        }
        Err(_) => {
            set_error("native ovfmi panic");
            default
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ovfmi_native_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_create(path: *const c_char) -> *mut c_void {
    guarded(std::ptr::null_mut(), || {
        let path = unsafe { string_arg(path, "path")? };
        Ok(Box::into_raw(Box::new(Host::new(Path::new(path))?)) as *mut c_void)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_set_real(
    handle: *mut c_void,
    name: *const c_char,
    value: f64,
) -> i32 {
    guarded(-1, || {
        unsafe { host_mut(handle)? }.set_real(unsafe { string_arg(name, "name")? }, value)?;
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_get_real(
    handle: *mut c_void,
    name: *const c_char,
    value: *mut f64,
) -> i32 {
    guarded(-1, || {
        if value.is_null() {
            return Err("output value must not be null".to_string());
        }
        unsafe { *value = host_mut(handle)?.get_real(string_arg(name, "name")?)? };
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_step(handle: *mut c_void, target_time: f64) -> i32 {
    guarded(-1, || {
        unsafe { host_mut(handle)? }.step(target_time)?;
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_reset(handle: *mut c_void) -> i32 {
    guarded(-1, || {
        unsafe { host_mut(handle)? }.reset()?;
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ovfmi_native_destroy(handle: *mut c_void) {
    guarded((), || {
        if !handle.is_null() {
            drop(unsafe { Box::from_raw(handle as *mut Host) });
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_model(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../demoapp/usd")
            .join(name)
    }

    // These execute in a Rust test binary, not a Python process. Build the
    // example FMUs first with demoapp/build_fmu.py on the current platform.
    #[test]
    fn fmi3_runs_and_resets_without_python() {
        let mut host = Host::new(&demo_model("BouncingBall.fmu3")).unwrap();
        for _ in 0..2 {
            host.set_real("h", 2.0).unwrap();
            host.step(0.1).unwrap();
            assert!((1.9..2.0).contains(&host.get_real("h").unwrap()));
            assert!(host.step(0.1).is_err());
            assert!(host.step(f64::NAN).is_err());
            host.reset().unwrap();
        }
    }

    #[test]
    fn ssp_container_runs_without_python() {
        let mut host = Host::new(&demo_model("conveyor_demo.native.fmu")).unwrap();
        host.set_real("operatorSpeed", 16.0).unwrap();
        host.set_real("rejectSpeed", 16.0).unwrap();
        host.set_real("enable", 1.0).unwrap();
        for i in 1..=600 {
            host.step(i as f64 / 60.0).unwrap();
        }
        for i in 0..5 {
            assert!((host.get_real(&format!("targetVelocity{i}")).unwrap() - 16.0).abs() < 1e-5);
        }
    }
}
