# Security Framework for GenAI Applications

This directory contains security configurations, policies, and tools to secure your GenAI application.

## Directory Structure

```
8_security/
├── iam/                    # Identity and Access Management
│   ├── rbac/               # Role-Based Access Control policies
│   ├── service-accounts/   # Service account configurations
│   └── oidc/               # OpenID Connect configurations
├── network/                # Network security
│   ├── policies/           # Network policies
│   └── vpc/                # VPC and network configurations
├── secrets/                # Secrets management
│   ├── vault/              # HashiCorp Vault configurations
│   └── k8s/                # Kubernetes secrets management
├── scanning/               # Security scanning tools
│   ├── trivy/              # Container scanning
│   ├── bandit/             # Python code scanning
│   └── tfsec/              # Terraform security scanning
├── compliance/             # Compliance frameworks
│   ├── gdpr/               # GDPR compliance
│   ├── hipaa/              # HIPAA compliance
│   └── soc2/               # SOC 2 compliance
├── monitoring/             # Security monitoring
│   ├── falco/              # Runtime security monitoring
│   └── osquery/            # Endpoint monitoring
└── policies/               # Security policies
    ├── pod-security/       # Pod security policies
    ├── network-policy/     # Network policies
    └── psp/                # Pod Security Policies (deprecated in 1.25+)
```

## Security Best Practices

### 1. Authentication & Authorization
- **OAuth 2.0/OIDC**: Implement for user authentication
- **JWT Validation**: For API authentication
- **RBAC**: Fine-grained access control
- **MFA**: Enforce for all admin access

### 2. Network Security
- **Network Policies**: Restrict pod-to-pod communication
- **Service Mesh**: Use Istio or Linkerd for mTLS
- **WAF**: Web Application Firewall for API protection
- **DDoS Protection**: Enable cloud provider DDoS protection

### 3. Secrets Management
- **Vault**: Central secrets management
- **KMS**: Key management for encryption
- **Secrets Rotation**: Automate secrets rotation

### 4. Container Security
- **Image Scanning**: Scan for vulnerabilities in container images
- **Rootless Containers**: Run containers as non-root
- **Read-only Filesystems**: Where possible
- **Resource Limits**: Set CPU/memory limits

### 5. Monitoring & Auditing
- **SIEM Integration**: Centralized logging
- **Audit Logs**: Enable Kubernetes audit logging
- **Falco**: Runtime security monitoring

## Quick Start

### 1. Set up Vault
```bash
cd 8_security/secrets/vault
vault server -config=vault-config.hcl
```

### 2. Apply Network Policies
```bash
kubectl apply -f 8_security/network/policies/
```

### 3. Run Security Scans
```bash
# Container scanning
trivy image your-image:tag

# Python code scanning
bandit -r ./src

# Terraform scanning
tfsec ./
```

## Compliance

### GDPR
- Data encryption at rest/transit
- Right to be forgotten
- Data portability
- DPO contact information

### HIPAA
- ePHI protection
- Access controls
- Audit logging
- Business Associate Agreements (BAAs)

### SOC 2
- Security policies
- Change management
- Incident response
- Vendor management

## Incident Response

1. **Detection**: Monitor for security events
2. **Containment**: Isolate affected systems
3. **Eradication**: Remove the threat
4. **Recovery**: Restore services
5. **Post-mortem**: Document and learn

## Security Tools

| Tool | Purpose |
|------|---------|
| **Vault** | Secrets management |
| **Trivy** | Container scanning |
| **Falco** | Runtime security |
| **OPA** | Policy enforcement |
| **Anchore** | Container analysis |
| **kube-bench** | K8s CIS benchmark |
| **kube-hunter** | K8s penetration testing |

## Contributing

1. Report security issues to security@your-org.com
2. Follow secure coding practices
3. Keep dependencies updated
4. Regular security training

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
