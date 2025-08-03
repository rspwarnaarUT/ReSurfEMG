## Reporting Security Vulnerabilities

If you discover a security vulnerability in ReSurfEMG, we kindly ask you to report it responsibly.

- Please **do not** open a public issue.
- Instead, email us at  e.mos-oppersma@utwente.nl with details.
- We will acknowledge your report within 14 business days and aim to resolve issues promptly.

Thank you for helping keep ReSurfEMG secure!

## Secure Development Knowledge

At least one core developer is familiar with common Python security vulnerabilities, including:

- Code injection and deserialization risks
- Command and path injection
- Dependency and secret management
- Secure authentication and cryptographic practices

We actively apply mitigation strategies such as input validation, secure coding patterns, and dependency auditing to reduce risk.

## Use of Cryptography


This project is not a cryptographic library. When cryptographic functionality is required (e.g., hashing, encryption), we rely exclusively on well-established, FLOSS cryptographic libraries such as:

- `cryptography`
- `hashlib`
- `secrets`

We do not implement custom cryptographic algorithms or primitives. This ensures our use of cryptography is secure, reliable, and aligned with best practices.


This project uses cryptographic functions (e.g., SHA-256) for file integrity checking. These functions are implemented using Python’s standard `hashlib` library, which is a well-established and FLOSS cryptographic tool. We do not implement custom cryptographic algorithms.
