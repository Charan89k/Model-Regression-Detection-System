import json
import random
from pathlib import Path
from uuid import uuid4


def generate_dataset():
    topics = [
        (
            "password_reset",
            [
                "I forgot my password",
                "Reset password link not working",
                "Can't login to my dashboard",
            ],
        ),
        (
            "billing_refund",
            ["Double charged this month", "Refund my pro subscription", "Invoice #{} is incorrect"],
        ),
        (
            "api_outage",
            [
                "API returning 500 Internal Server Error",
                "Webhook failing constantly",
                "Rate limit exceeded unexpectedly",
            ],
        ),
        (
            "feature_request",
            [
                "Please add dark mode",
                "Support SAML SSO for enterprise",
                "We need more CI/CD integrations",
            ],
        ),
    ]

    cases = []
    # Generate exactly 80 realistic emails
    for i in range(80):
        topic_id, subjects = random.choice(topics)
        subject_template = random.choice(subjects)
        subject = subject_template.format(random.randint(1000, 9999))

        # Determine difficulty and craft the email body
        diff_roll = random.random()
        if diff_roll < 0.4:
            difficulty = "easy"
            body = (
                f"Hi,\n\nI am having an issue: {subject}. "
                f"Can you please help me fix this as soon as possible?\n\nThanks,\nUser_{i}"
            )
        elif diff_roll < 0.7:
            difficulty = "medium"
            body = (
                f"Hello support team,\n\nI've been trying to deal with '{subject}' but it's "
                "really confusing. I am on the Pro tier and this should be working. "
                "Can someone look into this and let me know the status?\n\nRegards,\nJohn Doe"
            )
        elif diff_roll < 0.9:
            difficulty = "hard"
            body = (
                f"URGENT: Regarding {subject}.\n\nThis is blocking our production launch. "
                "I tried clearing my cache, reviewing the documentation, and even testing "
                "on a different network, but the problem persists. We need an engineer "
                "to review our account logs immediately."
            )
        else:
            difficulty = "expert"
            body = (
                f"We are seeing an anomaly loosely related to '{subject}'. However, upon "
                "inspecting our egress proxy logs, we noticed a 502 Bad Gateway originating "
                "from your edge nodes specifically when we send a payload exceeding 2MB "
                "over a kept-alive HTTP/2 connection. Is there a known MTU issue or WAF "
                "rule dropping this?\n\nBest,\nStaff Engineer"
            )

        cases.append(
            {
                "id": f"case-{uuid4().hex[:8]}",
                "difficulty": difficulty,
                "variables": {
                    "email_subject": subject,
                    "email_body": body,
                    "customer_tier": random.choice(["free", "pro", "enterprise"]),
                },
                "expected_output": topic_id,
                "evaluation_criteria": [f"Must classify the intent as {topic_id}"],
            }
        )

    dataset = {
        "metadata": {
            "name": "support_routing",
            "version": "1.0",
            "description": "Customer support emails for automated routing.",
        },
        "cases": cases,
    }

    out_dir = Path("datasets/support_routing")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v1.0.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("Generated datasets/support_routing/v1.0.json with 80 cases.")


if __name__ == "__main__":
    generate_dataset()
