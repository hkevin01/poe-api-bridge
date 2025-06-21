from modal import Image, App, asgi_app, Secret
from server import app as fastapi_app

# Create a Stub object - this is the main entry point for Modal
app = App("poe-api-bridge")

# Define requirements
REQUIREMENTS = [
    "fastapi==0.115.6",
    "fastapi-poe==0.0.56",
    "tiktoken==0.6.0"
]

# Create a custom image with production dependencies using new API
image = (
    Image.debian_slim(python_version="3.12")
    .pip_install(*REQUIREMENTS)
    .add_local_dir("static", "/root/static")
)

@app.cls(
    image=image,
    secrets=[Secret.from_dotenv()]
)
class PoeApiBridge:
    @asgi_app()
    def fastapi_app(self):
        return fastapi_app
