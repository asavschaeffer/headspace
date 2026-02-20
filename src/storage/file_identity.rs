//! Cross-platform file identity extraction for stable document tracking.

use std::path::Path;

use eyre::{Result, WrapErr};
use file_id::FileId;

/// Computes a stable, cross-platform file identity string.
///
/// On Unix this is `(device,inode)`, on Windows `(volume,file-id)`.
pub fn file_key(path: &Path) -> Result<String> {
    let id = file_id::get_file_id(path)
        .wrap_err_with(|| format!("failed to read file id: {}", path.display()))?;

    let key = match id {
        FileId::Inode {
            device_id,
            inode_number,
        } => format!("inode:{device_id}:{inode_number}"),
        FileId::LowRes {
            volume_serial_number,
            file_index,
        } => format!("lowres:{volume_serial_number}:{file_index}"),
        FileId::HighRes {
            volume_serial_number,
            file_id,
        } => format!("highres:{volume_serial_number}:{file_id}"),
    };

    Ok(key)
}

/// Fallback identity when OS-level file id cannot be read.
pub fn fallback_file_key(path: &Path) -> String {
    format!(
        "path:{}",
        path.to_string_lossy().replace('\\', "/").to_lowercase()
    )
}
