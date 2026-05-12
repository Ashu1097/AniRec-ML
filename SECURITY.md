# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| v22.x   | :white_check_mark: |
| < v22   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report security issues by emailing the maintainers directly at `security@anirec.example.com` with:

- A clear description of the vulnerability
- Steps to reproduce or proof-of-concept code
- Potential impact assessment
- Any suggested mitigations

You will receive a response within 72 hours. Once a fix is confirmed we will:
1. Prepare a patch release
2. Credit you in the changelog (unless you prefer to remain anonymous)
3. Publish a GitHub Security Advisory

## Scope

The following are **in scope**:
- Remote code execution via the FastAPI routes
- SQL injection via the feedback SQLite store
- Sensitive data exposure (API keys, credentials)
- Dependency vulnerabilities in `requirements.txt`

The following are **out of scope**:
- Issues requiring physical access to the server
- Social engineering attacks
- Vulnerabilities in third-party APIs (AniList, IMDb, TMDB, MovieLens)
