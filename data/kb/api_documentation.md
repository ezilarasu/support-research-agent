# API Documentation

## Overview

The REST API follows standard HTTP conventions. All requests and responses use JSON.
Base URL: `https://api.example.com/v1`

## Authentication

Include your API key in every request header:
```
Authorization: Bearer YOUR_API_KEY
```

API keys are created in the dashboard under Settings > API Keys.
Never commit API keys to source control — use environment variables.

## Pagination

All list endpoints return paginated results. Use `limit` and `offset` query parameters:

```
GET /v1/resources?limit=50&offset=0
```

Response format:
```json
{
  "data": [...],
  "total": 1234,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

## Resources

### Tickets

#### Create a ticket
```
POST /v1/tickets
Content-Type: application/json

{
  "subject": "Cannot access my account",
  "body": "I get a 401 error when I try to log in.",
  "priority": "high",
  "category": "technical"
}
```

Response (201 Created):
```json
{
  "id": "tkt_abc123",
  "subject": "Cannot access my account",
  "status": "open",
  "priority": "high",
  "category": "technical",
  "created_at": "2026-01-15T10:30:00Z",
  "assigned_to": null
}
```

#### Get a ticket
```
GET /v1/tickets/{ticket_id}
```

#### List tickets
```
GET /v1/tickets?status=open&priority=high&limit=20
```

Query parameters:
- `status`: open | in_progress | resolved | closed
- `priority`: low | medium | high | critical
- `category`: billing | technical | general
- `assigned_to`: agent ID (filter by assignee)
- `created_after`: ISO 8601 datetime

#### Update a ticket
```
PATCH /v1/tickets/{ticket_id}
Content-Type: application/json

{
  "status": "resolved",
  "resolution_note": "Issued refund for double charge"
}
```

#### Add a comment to a ticket
```
POST /v1/tickets/{ticket_id}/comments
Content-Type: application/json

{
  "body": "I've investigated the charge and will process a refund.",
  "is_internal": false
}
```

### Users

#### Get current user
```
GET /v1/me
```

Response:
```json
{
  "id": "usr_xyz789",
  "email": "user@company.com",
  "name": "Jane Smith",
  "plan": "pro",
  "created_at": "2025-03-01T00:00:00Z",
  "api_calls_used": 12450,
  "api_calls_limit": 50000
}
```

#### Update user profile
```
PATCH /v1/me
Content-Type: application/json

{
  "name": "Jane D. Smith",
  "timezone": "America/New_York"
}
```

### Billing

#### Get billing summary
```
GET /v1/billing/summary
```

Response:
```json
{
  "plan": "pro",
  "billing_period_start": "2026-01-01T00:00:00Z",
  "billing_period_end": "2026-02-01T00:00:00Z",
  "amount_due": 49.00,
  "currency": "USD",
  "next_billing_date": "2026-02-01T00:00:00Z"
}
```

#### List invoices
```
GET /v1/billing/invoices
```

#### Request a refund
```
POST /v1/billing/refunds
Content-Type: application/json

{
  "invoice_id": "inv_123",
  "amount": 49.00,
  "reason": "Service was unavailable during billing period"
}
```

## Webhooks

Configure webhooks to receive real-time event notifications.

### Supported events
- `ticket.created` — new ticket submitted
- `ticket.updated` — ticket status/priority changed
- `ticket.resolved` — ticket marked resolved
- `refund.approved` — refund request approved
- `refund.rejected` — refund request rejected
- `user.plan_changed` — subscription upgraded or downgraded

### Webhook payload format
```json
{
  "event": "ticket.created",
  "timestamp": "2026-01-15T10:30:00Z",
  "data": {
    "ticket_id": "tkt_abc123",
    "subject": "Cannot access my account"
  }
}
```

### Verifying webhook signatures
Every webhook includes a `X-Webhook-Signature` header. Verify it:
```python
import hmac, hashlib

def verify(payload: bytes, signature: str, secret: str) -> bool:
    computed = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={computed}", signature)
```

## Rate Limits

| Plan | Requests/minute | Requests/day |
|------|----------------|--------------|
| Free | 10 | 1,000 |
| Pro | 100 | 50,000 |
| Enterprise | 1,000 | Unlimited |

Rate limit headers in every response:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1737034260
```

## SDK Quick Reference

### Python
```python
from support_sdk import Client

client = Client(api_key="YOUR_KEY")

# Create ticket
ticket = client.tickets.create(
    subject="My invoice is wrong",
    body="I was charged twice this month.",
    priority="high",
    category="billing"
)

# List open tickets
for ticket in client.tickets.list(status="open"):
    print(ticket.id, ticket.subject)
```

### JavaScript/TypeScript
```typescript
import { SupportClient } from '@company/support-sdk';

const client = new SupportClient({ apiKey: process.env.API_KEY });

const ticket = await client.tickets.create({
  subject: 'My invoice is wrong',
  body: 'I was charged twice.',
  priority: 'high',
  category: 'billing',
});
```

## Changelog

### v1.3.0 (2026-01-01)
- Added webhook signature verification
- New `/v1/billing/refunds` endpoint
- Improved rate limit headers

### v1.2.0 (2025-10-01)
- Added ticket comments endpoint
- Pagination added to all list endpoints
- Deprecated v1/support/tickets (use /v1/tickets)

### v1.1.0 (2025-07-01)
- Added priority field to tickets
- Category filtering in ticket list endpoint
