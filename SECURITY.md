# Security Policy

*Updated: 2025-09-01 - Enhanced testing framework security and code quality
improvements*

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported | | ------- | ------------------ | | 0.7.2+ | ✅ Yes (Enhanced
Testing & Security) | | 0.7.2+ | ✅ Yes (Unified System) | | 0.7.x | ✅ Yes | | 0.6.x | ❌
Legacy Support Only | | < 0.6.0 | ❌ No |

## Security Features

### Automated Security Scanning

- **Bandit**: Continuous security vulnerability scanning for Python code
- **pip-audit**: Dependency vulnerability scanning
- **Pre-commit hooks**: Automatic security checks on every commit
- **GitHub Actions**: Security scanning in CI/CD pipeline

### Security Configuration

- **Zero medium/high severity issues** as of v0.7.2 (Enhanced Testing Framework)
- **Safe coding practices**: No hardcoded secrets, secure file operations
- **Dependency management**: Regular updates and vulnerability monitoring
- **Unified system security**: Safe shell integration and GPU environment setup
- **Shell completion security**: Sandboxed command completion without shell injection
  risks
- **Environment isolation**: Virtual environment integration without system
  contamination
- **GPU security**: Safe CUDA library path configuration without privilege escalation
- **Testing framework security**: Comprehensive test isolation and marker-based
  execution control

### Security Best Practices

1. **No hardcoded credentials**: All sensitive data is externalized
1. **Safe file operations**: Proper path validation and sanitization
1. **Input validation**: All user inputs are validated and sanitized
1. **Dependency isolation**: Optional dependencies are properly compartmentalized
1. **Cross-platform security**: Consistent security across Windows, macOS, and Linux
1. **Unified system security**:
   - Safe virtual environment integration (mamba, conda, venv support)
   - Secure shell completion without injection risks
   - GPU environment setup with proper privilege separation
   - System validation with controlled environment access
   - Enhanced testing framework with isolation controls

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue:

### For Critical Security Issues

**DO NOT** create a public GitHub issue for critical security vulnerabilities.

Instead, please:

1. **Email**: Send details to [wchen@anl.gov](mailto:wchen@anl.gov) with subject
   "SECURITY: Homodyne Vulnerability"
1. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Your contact information
1. **Expect**: Initial response within 48 hours
1. **Timeline**: Security fixes are prioritized and typically released within 7-14 days

### For Non-Critical Security Issues

For lower-impact security issues, you may:

1. Create a GitHub issue with the "security" label
1. Use our standard issue template
1. Include security impact assessment

## Security Response Process

1. **Acknowledgment**: We acknowledge receipt within 48 hours
1. **Assessment**: Security team reviews and assesses impact (1-3 days)
1. **Development**: Fix development and testing (3-7 days)
1. **Release**: Security patch release with advisory
1. **Disclosure**: Coordinated disclosure after fix is available

## Security Updates

Security updates are released as:

- **Patch releases** (e.g., 0.7.1 → 0.7.2) for security fixes
- **GitHub Security Advisories** for vulnerability notifications
- **Changelog entries** documenting security improvements

## Security Tools Integration

### Unified System Security Integration

```bash
# Install with security-focused development setup
pip install homodyne-analysis[dev]
homodyne-post-install --shell zsh --advanced
pre-commit install

# Validate security configuration
homodyne-validate --verbose
```

**Security features in unified system:**

- Safe shell completion with input sanitization
- Secure environment variable management
- GPU setup with proper privilege controls
- System validation without elevated access

### Manual Security Scanning

```bash
# Run security scans manually
bandit -r homodyne/ -f json -o bandit_report.json
pip-audit --desc --format=json --output=audit_report.json
safety check --json --output safety_report.json
```

### CI/CD Security

Our GitHub Actions include:

- Automated Bandit security scanning
- Dependency vulnerability checks
- Security regression testing
- SARIF upload to GitHub Security tab

## Vulnerability Disclosure Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment and initial assessment
- **Day 3-7**: Detailed analysis and fix development
- **Day 7-14**: Testing and validation
- **Day 14**: Security release and public disclosure
- **Day 15+**: Security advisory publication

## Security Contacts

- **Primary**: Wei Chen ([wchen@anl.gov](mailto:wchen@anl.gov))
- **Secondary**: GitHub Security Advisories
- **Repository**:
  [https://github.com/imewei/homodyne](https://github.com/imewei/homodyne)

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [GitHub Security Features](https://github.com/features/security)
- [Bandit Security Linter](https://bandit.readthedocs.io/)

______________________________________________________________________

**Note**: This security policy is reviewed and updated regularly. Last updated:
2025-09-01 (v0.7.2 - Enhanced Testing Framework Security)
