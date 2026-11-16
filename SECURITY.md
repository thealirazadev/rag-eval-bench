# Security policy

## Supported versions

Only the latest release (0.2.0, the version in `pyproject.toml`) on `main` receives security
fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
(Security tab, "Report a vulnerability") or email thealirazadev@gmail.com with the
steps to reproduce. You should get a reply within 72 hours.

## How rag-eval-bench protects API keys and data

- The Voyage AI key is read only from the `VOYAGE_API_KEY` environment variable and sent as a
  `Bearer` header to `api.voyageai.com`; it is never written to the config, the JSON output, or
  the Markdown report.
- Error messages for failed embedding calls report the HTTP status code and do not echo the key.
- The offline trigram embedder and BM25 retriever make no network calls and need no key.
- Dataset loading rejects a `relevant` id that does not exist in `documents.jsonl`, so malformed
  labels fail fast instead of producing silent scores.
