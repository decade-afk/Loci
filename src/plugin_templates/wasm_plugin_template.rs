


















































#![no_std]

use core::panic::PanicInfo;
use core::slice;


#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}


#[no_mangle]
    /// static function
pub static mut MEMORY: [u8; 65536] = [0; 65536]; 


#[no_mangle]
    /// extern function
pub extern "C" fn loci_initialize() -> i32 {
    
    0 
}


#[no_mangle]
    /// extern function
pub extern "C" fn loci_cleanup() -> i32 {
    
    0 
}


#[no_mangle]
    /// extern function
pub extern "C" fn loci_transform_logits(logits_ptr: i32, logits_len: i32) -> i32 {
    unsafe {
        
        let logits = slice::from_raw_parts_mut(
            logits_ptr as *mut f32,
            logits_len as usize,
        );

        
        if logits.len() > 0 {
            logits[0] += 1.5; 
        }
        if logits.len() > 1 {
            logits[1] -= 2.0; 
        }
    }

    0 
}


#[no_mangle]
    /// extern function
pub extern "C" fn loci_on_token_generated(
    token_id: i32,
    token_text_ptr: i32,
    token_text_len: i32,
) -> i32 {
    unsafe {
        
        let token_text = slice::from_raw_parts(
            token_text_ptr as *const u8,
            token_text_len as usize,
        );

        
        if token_text.len() >= 4 {
            if token_text[0] == b'S'
                && token_text[1] == b'T'
                && token_text[2] == b'O'
                && token_text[3] == b'P'
            {
                return 1; 
            }
        }
    }

    0 
}












