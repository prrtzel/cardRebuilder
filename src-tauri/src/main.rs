#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpListener;
use std::sync::Mutex;
use tauri::{Manager, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct AppState {
    child: Mutex<Option<CommandChild>>,
    port: Mutex<u16>,
}

fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("no free port")
        .local_addr()
        .unwrap()
        .port()
}

#[tauri::command]
fn get_backend_port(state: tauri::State<AppState>) -> u16 {
    *state.port.lock().unwrap()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            child: Mutex::new(None),
            port: Mutex::new(0),
        })
        .setup(|app| {
            let port = find_free_port();
            *app.state::<AppState>().port.lock().unwrap() = port;

            // ---- Build the sidecar command ----
            // Debug: run server.py directly from the project venv (hot-reload friendly).
            // Release: use the bundled externalBin. --port is the only contract.
            let sidecar_cmd = if cfg!(debug_assertions) {
                let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
                let root = manifest_dir.parent().unwrap();
                let python = root.join("venv/Scripts/python.exe");
                let server = root.join("python-backend/server.py");
                app.shell().command(python.to_string_lossy().into_owned()).args([
                    server.to_string_lossy().into_owned(),
                    "--port".into(),
                    port.to_string(),
                ])
            } else {
                app.shell()
                    .sidecar("primal-backend")
                    .expect("sidecar not found")
                    .args(["--port".into(), port.to_string()])
            };

            let (mut rx, child) = sidecar_cmd.spawn().expect("failed to spawn sidecar");
            *app.state::<AppState>().child.lock().unwrap() = Some(child);

            // =================================================================
            // PIPE DRAINER — this is the fix for the "app hangs during OCR" bug.
            // =================================================================
            // tauri-plugin-shell's internal reader threads push every stdout /
            // stderr line into `rx`. If we stop polling, the internal threads
            // block on `send()` once the channel's ~16 KB buffer fills. That
            // backpressures into their `read()` on the OS pipe, which
            // backpressures into Python's `write()`. EasyOCR, HuggingFace, and
            // PyTorch all log aggressively — OCR reliably fills that buffer
            // within seconds. The visible symptom is a silent hang with zero
            // errors. This loop must run for the entire lifetime of the child.
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let s = String::from_utf8_lossy(&line);
                            print!("[sidecar] {s}");
                        }
                        CommandEvent::Stderr(line) => {
                            let s = String::from_utf8_lossy(&line);
                            eprint!("[sidecar] {s}");
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[sidecar ERROR] {err}");
                        }
                        CommandEvent::Terminated(payload) => {
                            // Child died. Log so the user can see it in the
                            // terminal; the UI should also flip the model
                            // indicator to red on its next poll.
                            eprintln!(
                                "[sidecar] terminated: code={:?} signal={:?}",
                                payload.code, payload.signal
                            );
                            break;
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_backend_port])
        .on_window_event(|window, event| {
            // Kill the child when the last window closes. CloseRequested fires
            // *before* the window is destroyed, so the child is still alive and
            // the handle is still valid.
            if let WindowEvent::CloseRequested { .. } = event {
                if let Some(child) = window.state::<AppState>().child.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error running tauri app");
}