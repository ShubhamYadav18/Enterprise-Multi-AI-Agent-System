# IT Security Policy — Acme Corporation

## Access Control

### Authentication
- All employees must use **multi-factor authentication (MFA)** for all company systems.
- Passwords must be at least **14 characters** with a mix of uppercase, lowercase, numbers, and symbols.
- Passwords expire every **90 days** and cannot be reused within 12 cycles.
- Single Sign-On (SSO) via Okta is mandatory for all SaaS applications.

### Authorization
- Access follows the **principle of least privilege** — employees receive only the permissions necessary for their role.
- Elevated access requires manager approval and is reviewed quarterly.
- Third-party vendor access requires a signed NDA and is time-limited.

## Data Classification

| Level | Description | Examples |
|-------|-------------|----------|
| **Public** | Freely shareable | Marketing materials, blog posts |
| **Internal** | Company-wide access | Internal memos, org charts |
| **Confidential** | Need-to-know basis | Financial reports, client data |
| **Restricted** | Strict access controls | PII, credentials, source code |

## Incident Response

### Severity Levels
- **P1 (Critical)**: Active data breach, system compromise. Response within **15 minutes**.
- **P2 (High)**: Potential security vulnerability, suspicious activity. Response within **1 hour**.
- **P3 (Medium)**: Policy violation, minor vulnerability. Response within **4 hours**.
- **P4 (Low)**: General security inquiry. Response within **24 hours**.

### Reporting
Report security incidents immediately to **security@acmecorp.example.com** or via the **#security-incidents** Slack channel.

## Device Security

- All company laptops must have **full-disk encryption** enabled.
- Endpoint Detection and Response (EDR) software must be installed and running.
- Personal devices used for work must be enrolled in the **Mobile Device Management (MDM)** system.
- USB storage devices are **prohibited** on company machines.

## Compliance

Acme Corporation maintains compliance with:
- **SOC 2 Type II** — Audited annually.
- **ISO 27001** — Certified since 2019.
- **GDPR** — Full compliance for EU operations.
- **HIPAA** — Compliant for healthcare clients.
