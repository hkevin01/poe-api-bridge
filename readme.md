# Poe OpenAI Compatible API (Unofficial)

**Deployed app:** https://kamilio--poe-api-bridge-poeapibridge-fastapi-app.modal.run

Poe provides a [Python API](https://creator.poe.com/docs/external-application-guide), but it uses a custom format that requires custom providers, making it difficult to integrate into standard applications.

This project serves as a wrapper around the Poe API, providing a standard `/chat/completions` endpoint that can be used anywhere.

Additionally, it provides `/images/generations` and `/images/edits` endpoints that work with any bots capable of returning image attachments, but they also support audio and video bots.

## Endpoints

All bots work in all endpoints, and everything is multimodal. The endpoints differ in their intended user experience:

### `/chat/completions`
Provides a traditional **chat experience** where you interact conversationally with the bot. This endpoint is ideal for:
- Back-and-forth conversations
- Text-based interactions with multimodal capabilities. Models return URLs

### `/images/generations` and `/images/edits`
Provide a **work on file experience** where you submit content for processing or generation. These endpoints are ideal for:
- Image generation and editing
- File transformations
- Audio/video processing
- Single-request content creation

Both endpoint types support the same underlying bots and multimodal capabilities - the difference is in how they structure the interaction pattern for different use cases.

## Tool Calling

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

Take the token counting with a grain of salt - it uses tiktoken to count tokens. Poe doesn't report these numbers, so the counts might differ from actual tokens, especially on non-OpenAI models. For some models like image gen, the numbers won't make sense at all. 

## Supported Applications

I've tested Roo Code, Cline, and various other tools - they work reliably. Feel free to create an issue if something is broken.

# Future work
- Anthropic API - similar to https://github.com/1rgs/claude-code-proxy. It would be nice to use Claude Code with Poe. 
- Audio API, maybe elevenlabs compat. LLMs are having hard time understanding that `images/generations` endpoint returns audio/video.
- Native Tool Calling, as soon as Poe supports it.
- Embeddings `openai.embeddings.create` would be nice
- OpenAI responses API https://platform.openai.com/docs/api-reference/responses
- MCP Client

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