# General FAQ & Company Policies

## Account Management

### How do I reset my password?
1. Go to the login page and click "Forgot password"
2. Enter your registered email address
3. Check your inbox for a reset link (valid for 24 hours)
4. If you don't receive the email, check your spam folder
5. If the issue persists, contact support with your account ID

### How do I update my email address?
Email changes require verification of both the old and new address:
1. Go to Settings > Account > Email
2. Enter your new email and click "Send verification"
3. Click the link in the email sent to your OLD address to confirm the change request
4. Click the link in the email sent to your NEW address to complete the change

### Can I have multiple accounts?
Each person may have one personal account. For team access, use the Team Management features:
- Invite teammates from Settings > Team
- Assign roles: Admin, Editor, Viewer, Billing-only
- Team seats are billed per seat on the Pro plan

### How do I delete my account?
Account deletion is permanent and cannot be undone. To delete:
1. Export your data first: Settings > Data Export > Download All
2. Go to Settings > Account > Delete Account
3. Type "DELETE" to confirm
4. All data is purged within 30 days per our retention policy

## Subscription Plans

### What's included in each plan?

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| API calls/month | 1,000 | 50,000 | Unlimited |
| Team members | 1 | 10 | Unlimited |
| Data storage | 1 GB | 50 GB | Custom |
| Support | Community | Email (24h) | Dedicated SLA |
| Webhooks | No | Yes | Yes |
| SSO/SAML | No | No | Yes |

### How do I upgrade my plan?
1. Go to Settings > Billing > Plans
2. Click "Upgrade" on the desired plan
3. Enter payment details (or use existing card)
4. Upgrade is immediate — you get access right away

### Will I be charged immediately when upgrading?
Yes, upgrading charges the pro-rated amount for the current billing period immediately.
Your next regular billing date remains the same.

### What happens when I hit my API limit?
- Free plan: API calls return 429 errors until the next billing cycle
- Pro plan: You receive an email warning at 80% usage; at 100% you can purchase additional API calls or upgrade
- Enterprise: Contact your account manager for burst capacity

## Data & Privacy

### Where is my data stored?
All data is stored in US data centers by default.
EU customers on Enterprise plans can request EU-only data residency.

### How long is my data retained?
- Active accounts: indefinitely while the account is active
- Cancelled accounts: 90 days after cancellation, then permanently deleted
- Deleted accounts: 30 days after deletion request

### Is my data encrypted?
Yes:
- Data in transit: TLS 1.3
- Data at rest: AES-256 encryption
- API keys are stored as one-way hashes (we cannot recover your key — only reset it)

### Can I export my data?
Yes. Go to Settings > Data Export to download:
- Account information (JSON)
- API usage history (CSV)
- All created resources (JSON)

### Do you share my data with third parties?
We do not sell or share personal data. We use third-party processors (payment processor, cloud provider) under strict data processing agreements. See our Privacy Policy for the full list.

## Support & SLA

### How do I contact support?
- Free: Community forum only
- Pro: Email support at support@company.com — 24-hour response time
- Enterprise: Dedicated Slack channel + phone support — 4-hour SLA

### What's your uptime SLA?
- Pro: 99.5% uptime guarantee
- Enterprise: 99.9% uptime with financial credits for breaches

### How do I report a security vulnerability?
Email security@company.com with details. Do NOT post vulnerabilities publicly.
We offer a bug bounty program for qualifying disclosures.

## Frequently Asked Questions

**Q: How do I change my timezone for reports?**
A: Go to Settings > Preferences > Timezone. All timestamps in exports and dashboards will reflect the new timezone.

**Q: Can I use the service for commercial projects?**
A: Yes, all plans allow commercial use. Check the Terms of Service for restrictions.

**Q: How do I get a VAT invoice?**
A: Add your VAT number in Settings > Billing > Tax Information. Future invoices will include it. For past invoices, contact billing support.

**Q: Is there a free trial for Pro?**
A: Yes, new accounts get a 14-day Pro trial automatically. No credit card required.
