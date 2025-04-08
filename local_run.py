import argparse
import uvicorn
import os
from dotenv import load_dotenv


def load_config():
    # Load .env file
    load_dotenv()

    # Get environment variables with defaults
    return {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.environ["SERVER_PORT"]),
        "reload": False,
    }


def main():
    # Load configuration from .env
    config = load_config()

    # Setup argument parser with defaults from .env
    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=config["reload"],
        help="Enable auto-reload",
    )

    args = parser.parse_args()

    print(f"Starting server on {config['host']}:{config['port']}")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print(
        f"API documentation available at: http://{config['host']}:{config['port']}/docs"
    )

    uvicorn.run(
        "server:app",
        host=config["host"],
        port=config["port"],
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
