#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

image_path = "scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg"

if not os.path.exists(image_path):
    print(f"❌ Image file not found: {image_path}")
    exit(1)

with open(image_path, "rb") as image_file:
    response = client.images.edit(
        model="StableDiffusionXL",
        image=image_file,
        prompt="Make the image more colorful and vibrant",
        n=1,
        response_format="url",
    )

print(f"✅ Image edit: {response.data[0].url}")
assert response.data[0].url
