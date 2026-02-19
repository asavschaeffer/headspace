/// Configurable thresholds for lifecycle status classification.
pub struct StatusThresholds {
    pub draft_days: u64,
    pub active_days: u64,
    pub rot_days: u64,
    pub active_min_bytes: usize,
    pub rot_max_bytes: usize,
}

impl Default for StatusThresholds {
    fn default() -> Self {
        Self {
            draft_days: 14,
            active_days: 30,
            rot_days: 365,
            active_min_bytes: 2048,
            rot_max_bytes: 8192,
        }
    }
}

/// Determines lifecycle status with confidence.
#[allow(clippy::too_many_arguments)]
pub fn classify_status(
    rel_path: &str,
    name: &str,
    extension: &str,
    modified_at: u64,
    content_length: usize,
    topic_count: usize,
    entity_count: usize,
    now_epoch_secs: u64,
    thresholds: &StatusThresholds,
) -> (String, f32) {
    let path_lc = rel_path.to_ascii_lowercase();
    let name_lc = name.to_ascii_lowercase();
    let ext_lc = extension.to_ascii_lowercase();
    let age_secs = now_epoch_secs.saturating_sub(modified_at);
    let age_days = age_secs / 86_400;

    if has_draft_hint(&path_lc, &name_lc) {
        return ("Draft".to_string(), 0.93);
    }

    let is_authoring = matches!(ext_lc.as_str(), "md" | "txt" | "docx");
    if is_authoring && age_days <= thresholds.draft_days {
        return ("Draft".to_string(), 0.76);
    }

    if age_days <= thresholds.active_days && content_length >= thresholds.active_min_bytes {
        return ("Active".to_string(), 0.81);
    }

    let weak_signal = topic_count < 2 && entity_count < 2;
    if age_days > thresholds.rot_days && content_length < thresholds.rot_max_bytes && weak_signal {
        return ("Rot".to_string(), 0.79);
    }

    ("Reference".to_string(), 0.60)
}

fn has_draft_hint(path_lc: &str, name_lc: &str) -> bool {
    const TOKENS: [&str; 3] = ["draft", "wip", "todo"];
    TOKENS
        .iter()
        .any(|token| path_lc.contains(token) || name_lc.contains(token))
}

#[cfg(test)]
mod tests {
    use super::{StatusThresholds, classify_status};

    #[test]
    fn draft_from_name_keyword() {
        let (status, confidence) = classify_status(
            "notes/todo.md",
            "todo.md",
            "md",
            1_700_000_000,
            1_200,
            0,
            0,
            1_700_000_500,
            &StatusThresholds::default(),
        );
        assert_eq!(status, "Draft");
        assert!(confidence > 0.9);
    }

    #[test]
    fn active_recent_large() {
        let now = 1_700_000_000;
        let modified = now - (20 * 86_400);
        let (status, _) = classify_status(
            "project/spec.md",
            "spec.md",
            "md",
            modified,
            6_000,
            3,
            1,
            now,
            &StatusThresholds::default(),
        );
        assert_eq!(status, "Active");
    }

    #[test]
    fn rot_old_small_weak_signal() {
        let now = 1_700_000_000;
        let modified = now - (500 * 86_400);
        let (status, _) = classify_status(
            "archive/old.txt",
            "old.txt",
            "txt",
            modified,
            400,
            0,
            0,
            now,
            &StatusThresholds::default(),
        );
        assert_eq!(status, "Rot");
    }

    #[test]
    fn reference_fallback() {
        let now = 1_700_000_000;
        let modified = now - (100 * 86_400);
        let (status, _) = classify_status(
            "docs/guide.md",
            "guide.md",
            "md",
            modified,
            2_500,
            5,
            3,
            now,
            &StatusThresholds::default(),
        );
        assert_eq!(status, "Reference");
    }
}
