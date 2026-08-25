# Algerian Darija Text Corpus

A collection pipeline for building an open, text-only corpus of Algerian
Darija (Algerian Arabic dialect) — a low-resource language variety written
in both Arabic script and Arabizi (Latin script with numerals, e.g. `3=ع`,
`7=ح`, `9=ق`), and often mixed with French. The goal is a clean,
deduplicated dataset usable for general-purpose language model
pretraining.

## Data sources

| Source | Directory | Method |
|---|---|---|
| YouTube comments (Algerian channels: vlogs, comedy, sports, news, podcasts) | [`Youtube_scrap/`](Youtube_scrap) | YouTube Data API v3 |
| Djelfa.info forum (منتديات الجلفة) | [`Mountada_djelfa_scrap/`](Mountada_djelfa_scrap) | Direct scraping, robots.txt-compliant |

Each source is collected independently, then passed through a shared
pipeline: language/dialect filtering, deduplication, cleaning, and PII
anonymization, before being stored as structured JSONL.

## Repository layout

- `Youtube_scrap/` — YouTube comment scraper and config (seed channels/videos).
- `Mountada_djelfa_scrap/` — Djelfa.info forum scraper and config.
- `Data/` — combined, cleaned corpus output.
- `Notebooks/` — exploratory analysis and cleaning-rule development.

## Status

Active data collection. See each subdirectory's scripts for scraper usage.
