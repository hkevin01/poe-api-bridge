# OpenAI Image API Specification

## Image Generation
POST /v1/images/generations
- Input: Text prompt only
- Models: dall-e-3, dall-e-2
- Parameters: prompt, model, n, size, response_format

## Image Editing
POST /v1/images/edits
- Input: PNG file upload + text prompt
- Content-Type: multipart/form-data
- Parameters: image, prompt, mask, n, size, response_format

## Response Format
```json
{
  "created": 1589478378,
  "data": [{"url": "https://..."} | {"b64_json": "..."}]
}
```

## Poe Bridge Deviations
- Models: Accept any model name
- Size: Ignored, Poe bots determine dimensions
- Edits: No mask support

## Completions Extension (Poe Only)
POST /v1/completions
- Works like regular completions but returns image URL as text
- Accepts file uploads for image-to-image workflows


## Task
- images API 
  - Ignore any text and only return first file
- chat/completions API
  - stream or accumulate URLs as they are received
- Unit tests in the existing file - make sure to not make real requests
- verify_image_generation script - the URL should be returned
- generate image script script/....py
- Notes
  - For testing use Imagen-3-Fast
  - For testing edits use StableDiffusionXL
  - There's existing file scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg that you can use for testing

# Poe API
## Receiving Files from a Bot Response

If you called an image/video/audio generation bot (or any other bot that [outputs file attachments](https://creator.poe.com/docs/server-bots-functional-guides#sending-files-with-your-response)) on Poe, you will receive a [PartialResponse](https://creator.poe.com/docs/fastapi_poe-python-reference#fppartialresponse) object with the attachment in the `attachment` field for every file it outputs.

```python
class Imagen3Bot(fp.PoeBot):
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        
        sent_files = []
        async for msg in fp.stream_request(
            request, "Imagen-3", request.access_key
        ):
            # Add whatever logic you'd like to handle text responses from the Bot
            pass
            # If there is an attachment, add it to the list of sent files
            if msg.attachment:
                sent_files.append(msg.attachment)
		
        # This re-attaches the received files to this bot's output
        for file in sent_files:
            await self.post_message_attachment(
                message_id=request.message_id,
                download_url=file.url,
            )

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(server_bot_dependencies={"Imagen-3": 1})
```

```
class Attachment(BaseModel):
    url: str
    content_type: str
    name: str
    parsed_content: Optional[str] = None
  ```

In this example, we take the received files and attach them to the output. For more information on attaching files to your bot's output, see [Sending files with your response](https://creator.poe.com/docs/functional-guides-private#sending-files-with-your-response).