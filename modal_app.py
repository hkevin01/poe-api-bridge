from modal import Image, App, asgi_app, Secret, Mount
from server import app as fastapi_app
import os

# Create a Stub object - this is the main entry point for Modal
app = App("poe-api-bridge")

# Create a custom image with production dependencies
image = Image.debian_slim().pip_install_from_requirements("requirements-prod.txt")

# Get the path to the static directory relative to the current file
static_path = os.path.join(os.path.dirname(__file__), "static")


# The poe-api-bridge-secrets should contain:
# - LOGTAIL_SOURCE_TOKEN: Better Stack source token
# - LOGTAIL_SOURCE_ID: Identifier for the log source (e.g., "poe_api_bridge")
# - LOGTAIL_HOST: Better Stack ingestion host
@app.cls(
    image=image,
    secrets=[Secret.from_dotenv(), Secret.from_name("poe-api-bridge-secrets")],
    mounts=[Mount.from_local_dir(static_path, remote_path="/root/static")]
)
class PoeApiBridge:
    @asgi_app()
    def fastapi_app(self):
        return fastapi_app
