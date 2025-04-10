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
        "log_level": os.getenv("LOG_LEVEL", "info"),
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
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default=config["log_level"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Set log level in environment for the server to pick up
    os.environ["LOG_LEVEL"] = args.log_level

    print(f"Starting server on {config['host']}:{config['port']}")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print(f"Log level: {args.log_level}")
    print(
        f"API documentation available at: http://{config['host']}:{config['port']}/docs"
    )

    uvicorn.run(
        "server:app",
        host=config["host"],
        port=config["port"],
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
