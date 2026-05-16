---
scope: project
last-updated: YYYY-MM-DD
---

# Security Scanner — Persistent Memory

> **Auto-loaded by Claude Code when security-scanner agent is invoked.**
> Records confirmed vulnerabilities, cleared features, and security patterns for this repo.

---

## Cleared Security Reviews

<!-- Features that have passed security review — cleared = APPROVED -->
<!-- Format:
## [feature-name] — YYYY-MM-DD
- Cleared: YES/NO
- Critical findings: [count, resolved]
- PII handling: [assessment]
- Auth pattern: [confirmed correct/deviation noted]
-->

---

## Known Vulnerabilities / Open Issues

<!-- Active security issues discovered but not yet resolved -->
<!-- Format:
## [issue-name] — YYYY-MM-DD
**Severity:** CRITICAL/HIGH/MEDIUM/LOW
**Location:** [file:line]
**Description:** [what the issue is]
**Status:** OPEN / REMEDIATION IN PROGRESS / ACCEPTED RISK (with justification)
-->

---

## Security Patterns (verified for this repo)

<!-- Security patterns confirmed as correctly implemented in this codebase -->
<!-- Build confidence: record what's been verified to be working correctly -->
<!-- Example:
### Auth gate (verified 2026-04-12)
All HTTP endpoints have @AuthorisedPermission() or @UseGuards(JwtAuthGuard).
Verified via grep across all controllers — zero unguarded endpoints found.
-->

---

## Payment/PII Boundary

<!-- How PII and payment data flows in this repo — critical for security reviews -->
<!-- Example:
- BSB numbers: processed in NPP validators, never logged above DEBUG
- Account numbers: stored encrypted, returned masked in API responses
-->

---

## Memory Index
