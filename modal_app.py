from modal import Image, App, asgi_app
from server import app as fastapi_app

# Create a Stub object - this is the main entry point for Modal
app = App("poe-api-bridge")

# Create a custom image with production dependencies
image = Image.debian_slim().pip_install(
    "fastapi==0.115.6",
    "fastapi-poe==0.0.56",
    "tiktoken==0.6.0",
)


@app.cls(image=image, secrets=[])
class PoeApiBridge:
    @asgi_app()
    def fastapi_app(self):
        return fastapi_app
