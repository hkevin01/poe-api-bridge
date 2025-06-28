from modal import App, Image, Secret, asgi_app
from server import app as fastapi_app

# Create a Stub object - this is the main entry point for Modal
app = App("backend")


# Read requirements from requirements-prod.txt
def read_requirements():
    with open("requirements-prod.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


REQUIREMENTS = read_requirements()

# Create a custom image with production dependencies using new API
image = (
    Image.debian_slim(python_version="3.12")
    .add_local_file("requirements-prod.txt", "/root/requirements-prod.txt", copy=True)
    .pip_install(*REQUIREMENTS)
    .add_local_dir("static", "/root/static")
)


@app.cls(image=image, secrets=[Secret.from_dotenv()])
class PoeApiBridge:
    @asgi_app()
    def fastapi_app(self):
        return fastapi_app
