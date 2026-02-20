use std::collections::{HashMap, HashSet};

/// Extracts deterministic topics and entities for metadata fallback.
pub fn extract_topics_and_entities(text: &str, extension: &str) -> (Vec<String>, f32, Vec<String>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for token in tokenize(text) {
        if token.len() < 3 || is_stop_word(&token) || token.chars().all(char::is_numeric) {
            continue;
        }
        *counts.entry(token).or_insert(0) += 1;
    }

    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut topics: Vec<String> = ranked.into_iter().take(8).map(|(topic, _)| topic).collect();
    if topics.is_empty() {
        let ext = extension.to_ascii_lowercase();
        if !ext.is_empty() {
            topics.push(format!("{ext}-file"));
        }
    }

    let entities = extract_entities(text);
    let confidence = confidence_score(text, topics.len(), entities.len());
    (topics, confidence, entities)
}

fn tokenize(text: &str) -> Vec<String> {
    let mut current = String::new();
    let mut out = Vec::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let mut seen = HashSet::new();

    for token in text.split_whitespace() {
        let cleaned = token
            .trim_matches(|c: char| !c.is_alphanumeric())
            .trim()
            .to_string();
        if cleaned.len() < 3 {
            continue;
        }

        let starts_upper = cleaned.chars().next().is_some_and(char::is_uppercase);
        if !starts_upper {
            continue;
        }

        let lowered = cleaned.to_ascii_lowercase();
        if seen.insert(lowered) {
            entities.push(cleaned);
        }

        if entities.len() >= 8 {
            break;
        }
    }

    entities
}

#[allow(
    clippy::cast_precision_loss,
    reason = "Confidence score uses simple integer scaling"
)]
fn confidence_score(text: &str, topic_count: usize, entity_count: usize) -> f32 {
    let signal = text.chars().filter(|c| c.is_alphanumeric()).count();
    if signal < 20 {
        return 0.30;
    }
    let mut score = 0.45_f32;
    score += (topic_count.min(8) as f32) * 0.04;
    score += (entity_count.min(8) as f32) * 0.03;
    score.min(0.90)
}

fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "that"
            | "with"
            | "this"
            | "from"
            | "into"
            | "your"
            | "have"
            | "will"
            | "are"
            | "was"
            | "were"
            | "you"
            | "not"
            | "all"
            | "but"
            | "can"
            | "has"
            | "had"
            | "its"
            | "let"
            | "use"
            | "using"
            | "file"
            | "files"
            | "code"
            | "true"
            | "false"
            | "null"
    )
}

#[cfg(test)]
mod tests {
    use super::extract_topics_and_entities;

    #[test]
    fn extracts_topics_from_prose() {
        let text =
            "Headspace indexes filesystem documents and builds semantic search with embeddings.";
        let (topics, confidence, _) = extract_topics_and_entities(text, "md");
        assert!(!topics.is_empty());
        assert!(confidence > 0.4);
    }

    #[test]
    fn extracts_topics_from_code_like_text() {
        let text = "fn run_ingest(path: &Path) { let store = Store::load(path)?; }";
        let (topics, _, _) = extract_topics_and_entities(text, "rs");
        assert!(topics
            .iter()
            .any(|t| t == "run_ingest" || t == "store" || t == "path"));
    }

    #[test]
    fn handles_empty_text() {
        let (topics, confidence, entities) = extract_topics_and_entities("", "txt");
        assert_eq!(topics, vec!["txt-file".to_string()]);
        assert!(confidence <= 0.4);
        assert!(entities.is_empty());
    }
}
