"""Razorpay payments in INR (UPI, cards, net banking): an order per plan purchase, checked by signature.

Flow: the console asks POST /billing/checkout -> we create a Razorpay order for the plan's price -> Razorpay Checkout runs in
the browser -> it returns payment_id + order_id + signature -> POST /billing/verify checks
HMAC_SHA256(order_id|payment_id, key_secret) and starts the plan for 30 days. The webhook (payment.captured / order.paid) does the
same, so a plan starts even if the browser closes before /billing/verify; an order never starts a plan twice.

Keys: RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET (environment) or the owner dashboard. Test keys
(rzp_test_...) take test payments only.
"""
import hashlib
import hmac
import json

import httpx

API = "https://api.razorpay.com/v1"


class PaymentError(Exception):
    pass


def configured(keys):
    return bool(keys.get("key_id") and keys.get("key_secret"))


async def create_order(keys, amount_paise, receipt, notes):
    """A Razorpay order (dict with id, amount, currency)."""
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(API + "/orders", auth=(keys["key_id"], keys["key_secret"]),
                             json={"amount": int(amount_paise), "currency": "INR", "receipt": receipt[:40], "notes": notes})
    except httpx.HTTPError as e:
        raise PaymentError("cannot reach Razorpay: %s" % e)
    if r.status_code != 200:
        try: msg = r.json()["error"]["description"]
        except (ValueError, KeyError, TypeError): msg = "HTTP %d" % r.status_code
        raise PaymentError("Razorpay refused the order: %s" % msg)
    return r.json()


def payment_signature_ok(keys, order_id, payment_id, signature):
    want = hmac.new(keys["key_secret"].encode(), ("%s|%s" % (order_id, payment_id)).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, str(signature or ""))


def webhook_event(secret, raw_body, signature):
    """The parsed webhook body if its X-Razorpay-Signature is right, else None."""
    if not secret: return None
    want = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, str(signature or "")): return None
    try: return json.loads(raw_body)
    except ValueError: return None
