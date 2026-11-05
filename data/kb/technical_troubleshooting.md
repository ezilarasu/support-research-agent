# Technical Troubleshooting Guide

## API Integration Issues

### Authentication Failures (401 Unauthorized)
**Symptoms:** API returns 401 status code on every request.

**Common causes and fixes:**
1. **Expired API key** — generate a new key in your dashboard under Settings > API Keys
2. **Wrong header format** — use `Authorization: Bearer YOUR_API_KEY` (note the space after Bearer)
3. **Key not yet activated** — new keys take up to 60 seconds to propagate
4. **IP restriction** — check if the key is locked to specific IPs in your key settings

**Quick test:**
```bash
curl -H "Authorization: Bearer YOUR_KEY" https://api.example.com/v1/ping
```
Expected response: `{"status": "ok", "user": "your-email@company.com"}`

### Rate Limit Errors (429 Too Many Requests)
The API enforces rate limits per account tier:
- Free tier: 60 requests/minute
- Pro tier: 600 requests/minute
- Enterprise: custom limits

**Handling rate limits in code:**
```python
import time

def api_call_with_retry(endpoint, data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(endpoint, json=data)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            continue
        return response
    raise Exception("Max retries exceeded")
```

### Connection Timeouts
Default timeout is 30 seconds. Long operations (bulk exports, complex queries) may exceed this.

**Fix:** Set an explicit timeout matching the expected operation duration:
```python
response = requests.post(endpoint, json=data, timeout=120)  # 2 minute timeout
```

## SDK Installation Problems

### Python SDK
```bash
pip install support-sdk==2.1.0
```

**Dependency conflicts:** If you see version conflicts with `requests` or `httpx`:
```bash
pip install support-sdk==2.1.0 --upgrade-strategy eager
```

**Verify installation:**
```python
import support_sdk
print(support_sdk.__version__)  # Should print 2.1.0
```

### Node.js SDK
```bash
npm install @company/support-sdk@^2.0.0
```

**CommonJS vs ESM:** Use `require()` for CommonJS, `import` for ESM projects. Do not mix.

## Data Sync Issues

### Webhook Delivery Failures
If webhooks are not arriving:
1. Check the webhook URL is publicly accessible (not localhost)
2. Verify your server returns HTTP 200 within 5 seconds
3. Check the webhook signature validation — use the provided secret
4. Review delivery logs in Settings > Webhooks > Delivery History

**Webhook signature verification:**
```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

### Database Connection Errors
**Error:** `Connection refused: localhost:5432`

This usually means:
- Database service is not running: `sudo systemctl start postgresql`
- Wrong port in connection string
- Firewall blocking the port

**Connection string format:**
```
postgresql://username:password@host:5432/database_name
```

## Performance Issues

### Slow Query Resolution
Queries taking more than 2 seconds indicate:
- Missing database index — check `EXPLAIN ANALYZE` output
- N+1 query problem — use `include` parameters to eager-load relationships
- Result set too large — add pagination with `limit` and `offset`

### Memory Usage Spikes
If the application uses excessive memory:
1. Check for large result sets loaded into memory — use streaming/pagination
2. Look for memory leaks in long-running background jobs
3. Set `MAX_MEMORY_MB=512` in your `.env` to enforce a limit

## Error Codes Reference

| Code | Meaning | Action |
|------|---------|--------|
| 400 | Bad request — malformed input | Check request payload format |
| 401 | Unauthorized — invalid/expired key | Regenerate API key |
| 403 | Forbidden — insufficient permissions | Upgrade account tier |
| 404 | Not found — resource doesn't exist | Verify the resource ID |
| 409 | Conflict — duplicate resource | Check if resource already exists |
| 422 | Validation error — check error details | Fix the invalid fields |
| 429 | Rate limit exceeded | Back off and retry |
| 500 | Internal server error | Contact support with request ID |
| 503 | Service unavailable | Check status page, retry later |

## Frequently Asked Questions

**Q: My API key works in curl but not in my code — why?**
A: Check for extra whitespace around the key. Use `key.strip()` before passing to the SDK.

**Q: Can I use the API from a browser (CORS)?**
A: The API supports CORS for approved origins. Add your domain in Settings > CORS Configuration.

**Q: How do I debug API calls in production?**
A: Enable request logging with `LOG_LEVEL=DEBUG` and check the `X-Request-ID` header for tracing.
