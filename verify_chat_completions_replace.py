import requests
import json
import os
from dotenv import load_dotenv


def test_replace_response():
    # Load environment variables from .env file
    load_dotenv()

    # Try to get base URL from environment variable, fall back to local URL if not present
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    # Test data with a specific model and content as instructed
    test_data = {
        "model": "BetterDevBot",
        "messages": [
            {"role": "user", "content": "openai_test_case_replace_response"}
        ],
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Test streaming call
    print("\n=== Testing is_replace_response Functionality ===")
    accumulated_responses = []
    replacements_detected = 0
    last_content = ""
    full_content = ""
    found_final_response = False

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data,
            stream=True,
        ) as response:
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Streaming response chunks:")
                
                # Tracking variables
                current_chunk_content = ""
                
                for line in response.iter_lines():
                    if line:
                        # Remove the "data: " prefix if present and parse JSON
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]  # Remove "data: " prefix

                        # Skip [DONE] message
                        if line_text.strip() == "[DONE]":
                            print("\n[DONE] marker received")
                            continue

                        try:
                            chunk = json.loads(line_text)
                            # Print the raw chunk JSON for debugging
                            print(f"\n[DEBUG CHUNK] {json.dumps(chunk, indent=2)}")
                            
                            # Extract delta content
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            delta_content = delta.get("content", "")
                            
                            # Special handling for "Final Response"
                            if delta_content == "Final Response":
                                print("\n[DETECTED FINAL RESPONSE] This is the expected final content")
                                found_final_response = True
                                
                                # Save current response before replacing
                                if full_content:
                                    accumulated_responses.append(full_content)
                                
                                # Set content directly to the final response
                                full_content = "Final Response"
                                current_chunk_content = "Final Response"
                                print(delta_content, end="", flush=True)
                                continue
                            
                            # If we get content with a new role field, it's the first chunk of a message
                            role_field = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("role")
                            )
                            
                            # If we get a role field, it means we're starting a new message
                            if role_field:
                                # If we already had content, it means we're replacing a previous response
                                if full_content:
                                    print("\n[DETECTED REPLACEMENT] Previous content being replaced")
                                    replacements_detected += 1
                                    accumulated_responses.append(full_content)
                                    full_content = ""  # Reset the accumulated content
                                current_chunk_content = ""  # Reset for new message
                            
                            # If we're receiving a new content chunk without a role but after silence,
                            # it might be a replacement
                            if delta_content and not current_chunk_content and full_content:
                                print("\n[DETECTED REPLACEMENT] Previous content being replaced")
                                replacements_detected += 1
                                accumulated_responses.append(full_content)
                                full_content = ""  # Reset
                                
                            if delta_content:
                                current_chunk_content += delta_content
                                full_content += delta_content
                                print(delta_content, end="", flush=True)
                                
                            # If we get a chunk with role but no content, it's the first chunk
                            role_field = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("role")
                            )
                            if role_field:
                                current_chunk_content = ""  # Reset for new message
                                
                        except json.JSONDecodeError:
                            print(f"Could not decode JSON: {line_text}")

                # Store the final content
                if full_content and full_content != accumulated_responses[-1] if accumulated_responses else True:
                    accumulated_responses.append(full_content)
                
                last_content = accumulated_responses[-1] if accumulated_responses else ""
                
                print("\n\n=== Test Results ===")
                if replacements_detected > 0:
                    print(f"✅ Replacements detected: {replacements_detected}")
                else:
                    print("⚠️ No replacements detected in the response")
                    
                print(f"\nAccumulated responses: {len(accumulated_responses)}")
                if accumulated_responses:
                    for i, resp in enumerate(accumulated_responses):
                        if i < len(accumulated_responses) - 1:
                            print(f"\nResponse #{i+1} (replaced):")
                            print("-" * 40)
                            print(resp)
                            print("-" * 40)
                        else:
                            print(f"\nFinal response #{i+1}:")
                            print("-" * 40)
                            print(resp)
                            print("-" * 40)
                
                # Verify final content is as expected
                if found_final_response:
                    # Extract only the "Final Response" text to show as the final result
                    final_content = "Final Response"
                    print("\n✅ Test passed - Found 'Final Response' in the response stream")
                    print(f"Clean final content: '{final_content}'")
                    
                    # Update the last content for display
                    last_content = final_content
                else:
                    print("\n❌ Test failed - Final response 'Final Response' not found")
                    print(f"Actual final response: '{last_content}'")
                
            else:
                print(
                    f"\n❌ Streaming query failed with status code: {response.status_code}"
                )
                print("Response:")
                print(response.text)
    except Exception as e:
        print(f"\n❌ Error during streaming request: {e}")


if __name__ == "__main__":
    test_replace_response()