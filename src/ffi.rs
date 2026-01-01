//! Ffi Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::engine::{AIService, AIConfig, GenerateRequest};
use crate::agent::AgentSystem;
use crate::sysinfo::{SystemInfo, get_system_info};



pub const LOCI_SUCCESS: i32 = 0;
pub const LOCI_ERROR_NULL_POINTER: i32 = -1;
pub const LOCI_ERROR_MODEL_NOT_LOADED: i32 = -2;
pub const LOCI_ERROR_CONFIG_ERROR: i32 = -3;
pub const LOCI_ERROR_INFERENCE_ERROR: i32 = -4;
pub const LOCI_ERROR_IO_ERROR: i32 = -5;
pub const LOCI_ERROR_UNKNOWN: i32 = -99;




unsafe fn rust_string_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}


unsafe fn c_str_to_rust<'a>(s: *const c_char) -> Result<&'a str, &'static str> {
    if s.is_null() {
        return Err("Null pointer");
    }
    CStr::from_ptr(s).to_str().map_err(|_| "Invalid UTF-8")
}


unsafe fn set_error_message(error_out: *mut *mut c_char, message: String) {
    if !error_out.is_null() {
        *error_out = rust_string_to_c(message);
    }
}








#[no_mangle]
pub unsafe extern "C" fn loci_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}






#[no_mangle]
pub extern "C" fn loci_ai_service_new() -> *mut AIService {
    Box::into_raw(Box::new(AIService::new()))
}






#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_free(service: *mut AIService) {
    if !service.is_null() {
        drop(Box::from_raw(service));
    }
}











#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_update_config(
    service: *mut AIService,
    config_json: *const c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || config_json.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let json_str = match c_str_to_rust(config_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, format!("Invalid config string: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    match serde_json::from_str::<AIConfig>(json_str) {
        Ok(config) => {
            service_ref.update_config(config);
            LOCI_SUCCESS
        }
        Err(e) => {
            set_error_message(error_out, format!("JSON parse error: {}", e));
            LOCI_ERROR_CONFIG_ERROR
        }
    }
}










#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_load_model(
    service: *mut AIService,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    match service_ref.load_model() {
        Ok(_) => LOCI_SUCCESS,
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_INFERENCE_ERROR
        }
    }
}









#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_unload_model(service: *mut AIService) -> i32 {
    if service.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    service_ref.unload_model();
    LOCI_SUCCESS
}










#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_is_model_loaded(service: *mut AIService) -> i32 {
    if service.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    if service_ref.is_model_loaded() {
        1
    } else {
        0
    }
}












#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_generate(
    service: *mut AIService,
    request_json: *const c_char,
    response_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || request_json.is_null() || response_out.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let json_str = match c_str_to_rust(request_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, format!("Invalid request string: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    let request: GenerateRequest = match serde_json::from_str(json_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, format!("JSON parse error: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    match service_ref.generate(request) {
        Ok(response) => {
            match serde_json::to_string(&response) {
                Ok(json) => {
                    *response_out = rust_string_to_c(json);
                    LOCI_SUCCESS
                }
                Err(e) => {
                    set_error_message(error_out, format!("JSON serialize error: {}", e));
                    LOCI_ERROR_UNKNOWN
                }
            }
        }
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_INFERENCE_ERROR
        }
    }
}










#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_get_config(
    service: *mut AIService,
    config_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || config_out.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let config = service_ref.get_config();

    match serde_json::to_string(&config) {
        Ok(json) => {
            *config_out = rust_string_to_c(json);
            LOCI_SUCCESS
        }
        Err(_) => LOCI_ERROR_UNKNOWN,
    }
}






#[no_mangle]
pub extern "C" fn loci_agent_system_new() -> *mut AgentSystem {
    match AgentSystem::new() {
        Ok(system) => Box::into_raw(Box::new(system)),
        Err(_) => ptr::null_mut(),
    }
}


#[no_mangle]
pub unsafe extern "C" fn loci_agent_system_free(system: *mut AgentSystem) {
    if !system.is_null() {
        drop(Box::from_raw(system));
    }
}












#[no_mangle]
pub unsafe extern "C" fn loci_get_system_info(
    info_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if info_out.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    match get_system_info() {
        Ok(info) => {
            match serde_json::to_string(&info) {
                Ok(json) => {
                    *info_out = rust_string_to_c(json);
                    LOCI_SUCCESS
                }
                Err(e) => {
                    set_error_message(error_out, format!("JSON serialize error: {}", e));
                    LOCI_ERROR_UNKNOWN
                }
            }
        }
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_UNKNOWN
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_roundtrip() {
        unsafe {
            let original = "Hello, FFI!".to_string();
            let c_str = rust_string_to_c(original.clone());
            assert!(!c_str.is_null());

            let rust_str = c_str_to_rust(c_str).unwrap();
            assert_eq!(rust_str, original);

            loci_string_free(c_str);
        }
    }

    #[test]
    fn test_ai_service_lifecycle() {
        unsafe {
            let service = loci_ai_service_new();
            assert!(!service.is_null());

            let is_loaded = loci_ai_service_is_model_loaded(service);
            assert_eq!(is_loaded, 0);

            loci_ai_service_free(service);
        }
    }

    #[test]
    fn test_error_handling() {
        unsafe {
            let result = loci_ai_service_is_model_loaded(ptr::null_mut());
            assert_eq!(result, LOCI_ERROR_NULL_POINTER);
        }
    }
}
