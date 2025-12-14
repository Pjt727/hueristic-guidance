use anyhow::{Context, Result};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs::File;

use crate::grammar::VCmessage;

#[derive(Debug, Deserialize)]
struct PemazyreRow {
    #[serde(rename = "Category")]
    category: String,
    #[serde(rename = "Kind")]
    kind: String,
    #[serde(rename = "Description")]
    description: String,
    #[serde(rename = "VC Agent Message")]
    mlr_message: String,
    #[serde(rename = "VC Agent Message - URLs")]
    message: String,
}

pub fn load_pemazyre_responses(csv_path: &str) -> Result<Vec<VCmessage>> {
    let file =
        File::open(csv_path).with_context(|| format!("Failed to open CSV file: {}", csv_path))?;

    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t') // Tab-delimited file
        .flexible(true) // Allow records with varying number of fields
        .from_reader(file);

    let mut vc_messages = Vec::new();
    let mut skipped_count = 0;

    for (idx, result) in reader.deserialize().enumerate() {
        match result {
            Ok(record) => {
                let record: PemazyreRow = record;

                // Skip rows with N/A messages (conversation_continuer placeholders)
                if record.message == "N/A" || record.message.trim().is_empty() {
                    skipped_count += 1;
                    continue;
                }

                // Clean up the messages by removing placeholders
                let clean_message = record
                    .message
                    .trim()
                    .replace("{{conversation_continuer}}", "")
                    .trim()
                    .to_string();
                let clean_mlr_message = record
                    .mlr_message
                    .trim()
                    .replace("{{conversation_continuer}}", "")
                    .trim()
                    .to_string();

                vc_messages.push(VCmessage {
                    category: record.category.trim().to_string(),
                    kind: record.kind.trim().to_string(),
                    description: record.description.trim().to_string(),
                    mlr_message: clean_mlr_message,
                    message: clean_message,
                });
            }
            Err(e) => {
                // Log the error but continue processing
                eprintln!(
                    "Warning: Skipping row {} due to parse error: {}",
                    idx + 2,
                    e
                );
                skipped_count += 1;
                continue;
            }
        }
    }

    if vc_messages.is_empty() {
        anyhow::bail!("No valid messages found in CSV file");
    }

    println!(
        "Loaded {} valid messages, skipped {} rows",
        vc_messages.len(),
        skipped_count
    );

    Ok(vc_messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_pemazyre() {
        let messages = load_pemazyre_responses("data/pemazyre.csv").unwrap();
        assert!(!messages.is_empty());
        println!("Loaded {} messages", messages.len());

        // Print first few for inspection
        for (i, msg) in messages.iter().take(3).enumerate() {
            println!("{}: {:?}", i, msg);
        }
    }
}
