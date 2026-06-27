# Engineering Guide — Acme Corporation

## Development Standards

### Programming Languages
- **Backend**: Python 3.12+, with type hints required on all public functions.
- **Frontend**: TypeScript with React 18+.
- **Infrastructure**: Terraform for IaC, Docker for containerization.

### Code Quality
- All code must pass **linting** (Ruff for Python, ESLint for TypeScript) before merge.
- Test coverage must remain above **80%** for all modules.
- Security scanning is run on every pull request via Snyk.

## Code Review Process

1. Create a feature branch from `main` using the naming convention: `feature/JIRA-123-short-description`.
2. Open a Pull Request with a clear description, linked JIRA ticket, and relevant labels.
3. All PRs require **at least 2 approvals** from team members before merge.
4. Address all review comments before merging.
5. Use **squash and merge** to keep the commit history clean.

## CI/CD Pipeline

Our CI/CD pipeline runs on **GitHub Actions** and includes the following stages:

| Stage | Description | Duration |
|-------|-------------|----------|
| Lint | Code style and formatting checks | ~1 min |
| Unit Tests | Isolated unit tests | ~3 min |
| Integration Tests | API and database integration tests | ~8 min |
| Security Scan | Vulnerability scanning | ~2 min |
| Build | Docker image build | ~4 min |
| Deploy (Staging) | Automatic deployment to staging | ~3 min |
| Deploy (Prod) | Manual approval required | ~3 min |

## Architecture Principles

- **Microservices**: Services are independently deployable with well-defined API contracts.
- **Event-Driven**: Asynchronous communication via message queues (AWS SQS / RabbitMQ).
- **12-Factor App**: All services follow 12-factor app methodology.
- **Observability**: All services emit structured logs, metrics, and traces.

## On-Call Rotation

Engineering teams maintain a weekly on-call rotation. On-call engineers are expected to respond to **P1 incidents within 15 minutes** and **P2 incidents within 1 hour**. Runbooks are maintained in the internal wiki.
