
from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv("api_key.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def classify_zero_shot(text: str) -> dict:
  prompt = f"""Classify the sentiment of this airline customer review into one of three categories: negative, neutral, positive.
Return ONLY a valid JSON object with keys "sentiment" and "reasoning".
Review: "{text}"
"""

  completion = client.chat.completions.create(
      model="openai/gpt-oss-20b",  
      messages=[
          {
              "role": "system",
              "content": (
                  "You are an expert sentiment classifier. Output strictly"
                  " JSON."
              ),
          },
          {"role": "user", "content": prompt},
      ],
      response_format={"type": "json_object"},
      temperature=0.0,
  )
  return json.loads(completion.choices[0].message.content)


def classify_few_shot(text: str) -> dict:
  # Use an f-string directly with {text} so we don't use .format() on JSON curly braces
  prompt = f"""Classify the sentiment of airline customer reviews into: negative, neutral, positive.
Return ONLY a valid JSON object with keys "sentiment" and "reasoning".

Examples:
Review: "@United thanks for the smooth flight to Denver!"
Output: {{"sentiment": "positive", "reasoning": "Customer expresses gratitude for a smooth flight."}}

Review: "@JetBlue what time is flight 428 scheduled to depart?"
Output: {{"sentiment": "neutral", "reasoning": "Customer is asking a factual logistical question."}}

Review: "@AmericanAir my bag was lost and customer service was rude."
Output: {{"sentiment": "negative", "reasoning": "Customer reports a lost bag and poor service."}}

Now classify this review:
Review: "{text}"
Output:"""

  completion = client.chat.completions.create(
      model="openai/gpt-oss-20b",
      messages=[{"role": "user", "content": prompt}],
      response_format={"type": "json_object"},
      temperature=0.0,
  )
  return json.loads(completion.choices[0].message.content)

if __name__ == "__main__":
  sample = "@Delta absolute worst experience ever, my luggage is missing!"
  print("--- Zero-Shot Result ---")
  print(classify_zero_shot(sample))
  print("\n--- Few-Shot Result ---")
  print(classify_few_shot(sample))