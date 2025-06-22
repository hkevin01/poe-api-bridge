# Poe API Bridge

**Deployed app:** https://io--poe-api-bridge-poeapibridge-fastapi-app.modal.run

Poe provides a [Python API](https://creator.poe.com/docs/external-application-guide), but it uses a custom format that requires custom providers, making it difficult to integrate into standard applications.

This project serves as a wrapper around the Poe API, providing a standard `/chat/completions` endpoint that can be used anywhere.

Additionally, it provides `/images/generations` and `/images/edits` endpoints that work with any bots capable of returning image attachments, but they also support audio and video bots.

## Tools

Poe has limited support for tools with a non-conventional API requiring users to provide Python executors. Many applications require tool usage, so this project implements "fake" tool calling via prompting and XML parsing. In theory, this approach can work with any bot, even those that don't natively support tool calling. See the specification in [`docs/fake_tool_calling_spec.md`](docs/fake_tool_calling_spec.md) for more details.

## Token Counting

The bridge reports usage statistics in the standard OpenAI format:
```json
{
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

Take the token counting with a grain of salt - it uses tiktoken to count tokens. Poe doesn't report these numbers, so the counts might differ from actual tokens, especially on non-OpenAI models.

## Supported Applications

I've tested Roo Code, Cline, and various other tools - they work reliably. Feel free to create an issue if something is broken.

# Development

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