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
    model="Imagen-3-Fast",
    prompt="A beautiful sunset over mountains",
    n=1,
    response_format="url"
)

print(f"✅ Image generation: {response.data[0].url}")
assert response.data[0].url
