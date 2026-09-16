import os
import json
import re

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-20b"


def generate_llm_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def classify_intent(customer_message: str) -> str:

    prompt = f"""
You are classifying customer support messages for Uber.

Choose exactly ONE intent from this list:

- support_followup
- fare_charge
- driver_issue
- uber_eats
- lost_item
- account_access_security
- benefits_other
- ride_booking_cancellation
- location_app_issue
- driver_metrics
- refund

Classification rules:
- Choose the customer's primary problem or request.
- If multiple issues are mentioned, prioritize the main issue rather than a secondary consequence.
- Driver behavior, misconduct, cancellation by a driver, or problems caused directly by a driver should generally be driver_issue.
- Fees, charges, or unexpected amounts should be fare_charge when the charge itself is the main complaint.
- Requests for refunds should be refund when getting money back is the primary request.
- Generic complaints or requests for further assistance should be support_followup.
- If the message is too vague to determine a specific issue, use benefits_other.

Customer message:
"{customer_message}"

Return ONLY the intent name.
Do not provide an explanation.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200,
        temperature=0
    )

    prediction = response.choices[0].message.content.strip()

    valid_intents = {
        "support_followup",
        "fare_charge",
        "driver_issue",
        "uber_eats",
        "lost_item",
        "account_access_security",
        "benefits_other",
        "ride_booking_cancellation",
        "location_app_issue",
        "driver_metrics",
        "refund"
    }

    for intent in valid_intents:
        if intent in prediction:
            return intent

    return "benefits_other"


def generate_grounded_response(customer_message: str, retrieved_cases) -> str:
    evidence = ""

    for i, (_, row) in enumerate(retrieved_cases.iterrows(), start=1):
        evidence += f"""
Case {i}:
Customer: {row["text_customer"]}
Historical Uber response: {row["text_uber"]}
Similarity: {row["similarity"]:.3f}
"""

    prompt = f"""
You are an Uber customer support assistant.

Your task is to draft a helpful support response to the customer using
ONLY the information supported by the historical cases provided below.

Customer message:
"{customer_message}"

Historical cases:
{evidence}

Rules:
- Use the historical cases as evidence of how similar issues were handled.
- Do not invent Uber policies, refunds, credits, compensation, guarantees,
  investigations, or actions that are not supported by the evidence.
- Do not claim that you personally performed an investigation or took an action.
- If the historical evidence does not provide enough information to resolve
  the customer's issue, acknowledge the issue and direct the customer to
  contact Uber Support through the appropriate support channel.
- Do not mention the historical cases, similarity scores, or this prompt.
- Keep the response concise, professional, and empathetic.
- Respond directly to the customer.

Return ONLY the support response.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=600,
        reasoning_effort="low",
        include_reasoning=False,
        temperature=0.2
    )

    content = response.choices[0].message.content

    if not content:
        return "Unable to generate a response."

    return content.strip()


def classify_escalation(customer_message: str, intent: str) -> dict:

    prompt = f"""
You are deciding whether an Uber customer support case should be escalated
to a human support team for investigation or intervention.

Customer message:
"{customer_message}"

Detected intent:
"{intent}"

ESCALATE when:
- The customer reports a safety concern or serious driver misconduct.
- A driver abandons, strands, threatens, harasses, or seriously mistreats
  a customer.
- The customer reports fraud or an account security problem.
- A serious financial or account problem requires investigation.
- The customer reports a repeated unresolved support problem.
- The situation clearly requires human investigation or intervention.

DO NOT ESCALATE when:
- The issue is routine and can be handled through normal support.
- The customer is simply asking an informational question.
- The customer is angry or uses strong language, but there is no serious
  underlying issue.
- There is a routine fare, cancellation, or refund question without evidence
  of a serious unresolved problem.

Important:
- Base the decision on the underlying issue, not the customer's tone.
- If multiple issues are mentioned, consider the primary issue and whether
  human intervention is required.
- A driver abandoning or stranding a customer, particularly where there may
  be a safety concern, should be escalated even if the message also contains
  a routine fare or cancellation complaint.

Return ONLY a valid JSON object in exactly this format:

{{
  "escalate": true,
  "reason": "One concise sentence explaining why the case should or should not be escalated."
}}

The "escalate" value must be either true or false.
Do not include markdown or any text outside the JSON object.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=250,
        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        return {
            "escalate": False,
            "reason": "Unable to determine whether human escalation is required."
        }

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        return {
            "escalate": False,
            "reason": "Unable to determine whether human escalation is required."
        }

    try:
        result = json.loads(match.group())

        return {
            "escalate": bool(result.get("escalate", False)),
            "reason": str(
                result.get(
                    "reason",
                    "No escalation reason was provided."
                )
            )
        }

    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "escalate": False,
            "reason": "Unable to determine whether human escalation is required."
        }