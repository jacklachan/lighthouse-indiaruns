# Security Policy

## Data & privacy posture

Lighthouse is designed to be safe with sensitive recruiting data:

- **No secrets in the repository.** There are no API keys, tokens, or credentials
  committed. The rank-time pipeline makes **no network calls** and needs no credentials.
- **No candidate data leaves the machine at rank-time.** Ranking is deterministic
  `numpy`/`pandas` over locally precomputed artifacts. No candidate data is sent to any
  hosted LLM. Claude was used **offline only** — to parse the job description into a
  rubric and to author the proxy eval labels — never on candidate records at rank-time.
- **The 100K candidate pool is never committed.** It is grader-supplied and excluded via
  `.gitignore` (`data/`).

## Reporting a vulnerability

This is a hackathon submission, not a production service. If you find a security or
privacy issue, please open a GitHub issue (omit any sensitive data) or contact the team
maintainers listed in `submission_metadata.yaml`.

## Supported versions

The `main` branch is the only supported version.
