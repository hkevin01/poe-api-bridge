# Poe Code Chat Backend

Backend service that provides OpenAI-compatible API endpoints using the [poe-api-wrapper](https://github.com/snowby666/poe-api-wrapper) to access GPT-4, Claude, Gemini, and other models for free.

## Features

- ✅ **Free access** to premium models (GPT-4, Claude, Gemini, etc.)
- ✅ **OpenAI-compatible API** - works with existing VS Code extensions
- ✅ **Code context awareness** - automatically includes project context
- ✅ **Streaming responses** - real-time chat experience
- ✅ **Multiple model support** - Claude, GPT, Gemini, and more
- ✅ **Easy setup** - simple token configuration

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements-prod.txt
```

### 2. Get Poe Tokens

You need to get your Poe tokens from [poe.com](https://poe.com/):

1. Go to [poe.com](https://poe.com/) and sign in
2. Open Developer Tools (F12)
3. Go to **Application** > **Cookies** > **poe.com**
4. Copy the values for `p-b` and `p-lat` cookies

### 3. Setup Tokens

Run the interactive setup script:

```bash
python setup_tokens.py
```

Follow the prompts to enter your tokens.

### 4. Start the Server

```bash
python start.py
```

The server will be available at:
- **API**: http://127.0.0.1:8000
- **Docs**: http://127.0.0.1:8000/docs

### 5. Test the API

```bash
python test_api.py
```

## Available Models

| Model | Poe Model | Description |
|-------|-----------|-------------|
| `claude-3.5-sonnet` | `claude_3_igloo` | Claude 3.5 Sonnet (latest) |
| `claude-3-opus` | `claude_2_1_cedar` | Claude 3 Opus |
| `claude-3-sonnet` | `claude_2_1_bamboo` | Claude 3 Sonnet |
| `claude-3-haiku` | `claude_3_haiku` | Claude 3 Haiku (fast) |
| `gpt-4` | `beaver` | GPT-4 |
| `gpt-4o` | `gpt4_o` | GPT-4o |
| `gpt-4o-mini` | `gpt4_o_mini` | GPT-4o Mini |
| `gpt-3.5-turbo` | `chinchilla` | GPT-3.5 Turbo |
| `gemini-pro` | `gemini_pro` | Gemini Pro |
| `gemini-1.5-pro` | `gemini_1_5_pro_1m` | Gemini 1.5 Pro |

## API Endpoints

### Health Check
```http
GET /health
```

### List Models
```http
GET /v1/models
```

### Chat Completions
```http
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "claude-3-haiku",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant."
    },
    {
      "role": "user",
      "content": "Hello! Can you help me with my code?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

### Image Generation (Coming Soon)
```http
POST /v1/images/generations
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
# Required Poe tokens
POE_P_B_TOKEN=your_p_b_token_here
POE_P_LAT_TOKEN=your_p_lat_token_here

# Optional tokens for enhanced functionality
POE_FORMKEY=your_formkey_here
POE_CF_BM=your_cf_bm_token_here
POE_CF_CLEARANCE=your_cf_clearance_token_here
```

## VS Code Extension Integration

Your VS Code extension can use this backend by pointing to the local API:

```typescript
const response = await fetch('http://127.0.0.1:8000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'claude-3-haiku',
    messages: [
      { role: 'system', content: systemContext },
      { role: 'user', content: userMessage }
    ]
  })
});
```

## Troubleshooting

### "poe-api-wrapper not installed"
```bash
pip install poe-api-wrapper==1.6.0
```

### "Required Poe tokens not found"
Run the setup script:
```bash
python setup_tokens.py
```

### "Poe client not initialized"
Check that your tokens are valid and the `.env` file exists.

### Connection errors
Make sure the server is running:
```bash
python start.py
```

## Development

### Running with Auto-reload
```bash
python start.py
```

### Testing
```bash
python test_api.py
```

### Manual Server Start
```bash
python server.py
```

## Credits

- [poe-api-wrapper](https://github.com/snowby666/poe-api-wrapper) - Python wrapper for Poe.com
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Poe.com](https://poe.com/) - AI chat platform

## License

This project is licensed under the same license as your VS Code extension.