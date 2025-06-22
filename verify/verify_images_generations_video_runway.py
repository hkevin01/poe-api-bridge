#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

response = client.images.generate(
    model="Runway-Gen-4-Turbo",
    prompt="A beautiful sunset over mountains with clouds moving slowly across the sky.",
    n=1,
    response_format="url"
)

print(f"✅ Runway video: {response.data[0].url}")
assert response.data[0].url