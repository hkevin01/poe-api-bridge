# Poe API Bridge

OpenAI API compatible server that proxies requests to Poe.com.

## Server
Start in dev mode (with auto-reload)
```
make start-dev
```

Start in production mode
```
make start
```

## Testing
Run automated tests
```
make test
```

Run verification scripts
```
python3 verify_regular_query.py
```

## Deployment
Deploy to Modal
```
make deploy
```

## Utils
Clean artifacts
```
make clean
```
