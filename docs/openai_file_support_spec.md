# OpenAI File Support Implementation Spec

## Overview
Support for OpenAI's multimodal file capabilities including images and documents.

## Input Methods
1. **Base64 encoding**: Embed file as `data:image/jpeg;base64,{encoded_data}`
2. **URL reference**: Direct file URL for OpenAI to fetch
3. **File upload**: Use `/files` endpoint, then reference file ID

## Message Structure
```json
{
  "model": "gpt-4o",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "prompt"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```

## Supported Formats
- Images: JPEG, PNG, WebP, GIF (non-animated)
- Documents: PDF (with text extraction)
- Max file size: 20MB

## Poe API File Support

### File Upload Function
```python
pdf_attachment = fp.upload_file(open("draconomicon.pdf", "rb"), api_key=api_key)
message = fp.ProtocolMessage(role="user", content="Hello world", attachments=[pdf_attachment])
```

### Client Implementation
- Add `upload_file()` method to client
- Support file attachments in `ProtocolMessage`
- Handle file objects and convert to appropriate format
- Enable file input for bot query requests

## Deviations from OpenAI

### File Support Differences
- **File Types**: Support all file types that Poe API supports (not limited to OpenAI's format restrictions)
- **Upload Method**: No `/files` endpoint implementation - files handled directly through message attachments
- **Processing**: Files converted from Poe attachment format to OpenAI message format internally

## Implementation Requirements
- Handle multipart content arrays in message structure
- Support base64 encoding for local files
- Validate file formats and sizes before sending
- Preserve existing text-only message compatibility
- Add file upload capability to Poe API client
- Convert Poe attachments to OpenAI format