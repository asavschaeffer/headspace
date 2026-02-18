use super::PipelineKind;

/// Selects a pipeline from MIME type first, then extension fallback.
pub fn select_pipeline(mime: &str, ext: &str) -> PipelineKind {
    let mime = mime.to_ascii_lowercase();
    let ext = ext.to_ascii_lowercase();

    if is_text_mime(&mime) || is_text_extension(&ext) {
        return PipelineKind::Text;
    }
    if mime == "application/pdf" || ext == "pdf" {
        return PipelineKind::Pdf;
    }
    if mime.starts_with("audio/") || mime.starts_with("video/") || mime.starts_with("image/") {
        return PipelineKind::Media;
    }
    if is_media_extension(&ext) {
        return PipelineKind::Media;
    }

    PipelineKind::Unknown
}

fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/")
        || matches!(
            mime,
            "application/json"
                | "application/xml"
                | "application/x-yaml"
                | "application/yaml"
                | "application/toml"
                | "application/javascript"
                | "application/x-sh"
        )
}

fn is_text_extension(ext: &str) -> bool {
    matches!(
        ext,
        "txt"
            | "md"
            | "rs"
            | "py"
            | "js"
            | "ts"
            | "jsx"
            | "tsx"
            | "json"
            | "toml"
            | "yaml"
            | "yml"
            | "html"
            | "css"
            | "scss"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "go"
            | "java"
            | "sh"
            | "bat"
            | "ps1"
            | "xml"
            | "csv"
            | "log"
            | "cfg"
            | "ini"
            | "env"
            | "sql"
            | "rb"
            | "php"
            | "swift"
            | "kt"
            | "r"
            | "lua"
            | "pl"
            | "ex"
            | "exs"
            | "zig"
            | "nim"
            | "v"
            | "d"
            | "makefile"
            | "dockerfile"
            | ""
    )
}

fn is_media_extension(ext: &str) -> bool {
    matches!(
        ext,
        "png"
            | "jpg"
            | "jpeg"
            | "gif"
            | "bmp"
            | "webp"
            | "heic"
            | "mp3"
            | "wav"
            | "m4a"
            | "flac"
            | "ogg"
            | "mp4"
            | "mov"
            | "mkv"
            | "avi"
            | "webm"
    )
}

#[cfg(test)]
mod tests {
    use super::select_pipeline;
    use crate::cortex::PipelineKind;

    #[test]
    fn routes_text_plain() {
        assert_eq!(select_pipeline("text/plain", "txt"), PipelineKind::Text);
    }

    #[test]
    fn routes_markdown_by_extension() {
        assert_eq!(
            select_pipeline("application/octet-stream", "md"),
            PipelineKind::Text
        );
    }

    #[test]
    fn routes_pdf() {
        assert_eq!(select_pipeline("application/pdf", "pdf"), PipelineKind::Pdf);
    }

    #[test]
    fn routes_media() {
        assert_eq!(select_pipeline("audio/mpeg", "mp3"), PipelineKind::Media);
    }

    #[test]
    fn routes_unknown() {
        assert_eq!(
            select_pipeline("application/octet-stream", "bin"),
            PipelineKind::Unknown
        );
    }
}
