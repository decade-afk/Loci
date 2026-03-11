mod loci;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            loci::loci_health,
            loci::loci_info,
            loci::loci_generate
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
