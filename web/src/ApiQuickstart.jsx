import React, { useState, useEffect } from 'react'
import Tabs from './components/Tabs'
import CodeBlock from './components/CodeBlock'
import InlineCode from './components/InlineCode'
import { Beaker, Rocket, Wrench, Lock, Bot, HelpCircle } from 'lucide-react'
import OpenAI from 'openai'
import './ApiQuickstart.css'

// API base URL constant
const API_BASE_URL = 'https://kamilio--poe-api-bridge-poeapibridge-fastapi-app.modal.run/v1'

// ApiPlayground Component
function ApiPlayground({ defaultModel }) {
    const [isCollapsed, setIsCollapsed] = useState(false)
    const [apiKey, setApiKey] = useState('')
    const [apiResponses, setApiResponses] = useState({
        chat: '',
        models: '',
        stream: '',
        imageGen: '',
        imageEdit: '',
        imageChatGen: ''
    })
    const [isLoading, setIsLoading] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')
    const [selectedSnippet, setSelectedSnippet] = useState('chat')

    const codeSnippets = {
        chat: {
            name: "Chat Completion",
            description: "Basic chat completion request with system and user messages",
            code: `// Chat completion example
const apiKey = "${apiKey || 'YOUR_POE_API_KEY'}";
const baseUrl = "${API_BASE_URL}";

const response = await fetch(\`\${baseUrl}/chat/completions\`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': \`Bearer \${apiKey}\`
  },
  body: JSON.stringify({
    model: "${defaultModel}",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Tell me a short joke about programming." }
    ]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);`
        },
        stream: {
            name: "Stream Chat",
            description: "Stream a chat completion response",
            code: `// Streaming chat completion using OpenAI client
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: "${apiKey || 'YOUR_POE_API_KEY'}",
  baseURL: "${API_BASE_URL}",
  dangerouslyAllowBrowser: true,
});

console.log('Starting streaming response...');

const stream = await openai.chat.completions.create({
  model: "${defaultModel}",
  messages: [
    { role: "user", content: "Count from 1 to 5 slowly." }
  ],
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) {
    console.log(content);
  }
}

console.log('Stream completed!');`
        },
        imageGen: {
            name: "Image Generation",
            description: "Generate images using Imagen-3-Fast",
            code: `// Image generation using OpenAI client
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: "${apiKey || 'YOUR_POE_API_KEY'}",
  baseURL: "${API_BASE_URL}",
  dangerouslyAllowBrowser: true,
});

const response = await openai.images.generate({
  model: "Imagen-3-Fast",
  prompt: "A cute robot painting a landscape",
  n: 1,
});

console.log('Generated image:', response.data[0].url);`
        },
        imageEdit: {
            name: "Image Edit",
            description: "Edit an existing image using StableDiffusionXL",
            code: `// Image editing using OpenAI client
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: "${apiKey || 'YOUR_POE_API_KEY'}",
  baseURL: "${API_BASE_URL}",
  dangerouslyAllowBrowser: true,
});

// Fetch the image and convert to File object
const imageUrl = "https://pfst.cf2.poecdn.net/base/image/728cd697ec740e193d2ecdb1de750ba0a48656a6b9c79dfbfe949c7ff1e1a8db?w=1024&h=1024&pmaid=404413313";
const imageResponse = await fetch(imageUrl);
const imageBlob = await imageResponse.blob();
const imageFile = new File([imageBlob], "image.png", { type: "image/png" });

console.log('Original image:', imageUrl);

const response = await openai.images.edit({
  model: "StableDiffusionXL",
  image: imageFile,
  prompt: "Make this image look like a cartoon",
  n: 1,
});

console.log('Edited image:', response.data[0].url);`
        },
        imageChatGen: {
            name: "Image Gen via Chat",
            description: "You can use /chat/completions to generate images as well, but you need to do the URL parsing yourself. It works also for audio, video, etc.",
            code: `// Image generation via chat completion
const apiKey = "${apiKey || 'YOUR_POE_API_KEY'}";
const baseUrl = "${API_BASE_URL}";

const response = await fetch(\`\${baseUrl}/chat/completions\`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': \`Bearer \${apiKey}\`
  },
  body: JSON.stringify({
    model: "Imagen-3-Fast",
    messages: [
      { role: "user", content: "Generate an image of a futuristic city skyline at sunset" }
    ]
  })
});

const data = await response.json();
console.log('Generated image:', data.choices[0].message.content);`
        },
        models: {
            name: "List Models",
            description: "Fetch all available models from the API. These are only some of the popular models. Feel free to use any model on Poe https://poe.com/explore?category=Official",
            code: `// List available models
const apiKey = "${apiKey || 'YOUR_POE_API_KEY'}";
const baseUrl = "${API_BASE_URL}";

const response = await fetch(\`\${baseUrl}/models\`, {
  method: 'GET',
  headers: {
    'Authorization': \`Bearer \${apiKey}\`
  }
});

const data = await response.json();
console.log(\`Found \${data.data.length} available models:\`);
data.data.forEach(model => {
  console.log(\`- \${model.id}\`);
});`
        },
    };

    const snippetTabs = [
        { id: 'chat', label: 'Chat' },
        { id: 'stream', label: 'Chat Streaming' },
        { id: 'imageGen', label: 'Image Gen' },
        { id: 'imageEdit', label: 'Image Edit' },
        { id: 'imageChatGen', label: 'Chat Image Gen' },
        { id: 'models', label: 'Model List' },
    ];

    const executeCode = async () => {
        if (!apiKey) {
            setErrorMessage('Please enter your API key.');
            return;
        }

        setIsLoading(true);
        setErrorMessage('');

        try {
            const snippet = codeSnippets[selectedSnippet];

            // Create a fake console that captures output
            const capturedLogs = [];
            const fakeConsole = {
                log: (...args) => {
                    const logMessage = args.map(arg =>
                        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                    ).join(' ');
                    capturedLogs.push(logMessage);
                },
                error: (...args) => {
                    const logMessage = '[ERROR] ' + args.map(arg =>
                        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                    ).join(' ');
                    capturedLogs.push(logMessage);
                },
                warn: (...args) => {
                    const logMessage = '[WARN] ' + args.map(arg =>
                        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                    ).join(' ');
                    capturedLogs.push(logMessage);
                }
            };

            // Remove import statements from the code before execution
            const codeWithoutImports = snippet.code.replace(/^import\s+.*$/gm, '');

            // Create a new async function from the flat code and execute it
            const asyncFunction = new Function('console', 'fetch', 'OpenAI', `
                return (async () => {
                    ${codeWithoutImports}
                })();
            `);

            await asyncFunction(fakeConsole, fetch, OpenAI);

            // Store the captured logs as the response for the current tab
            setApiResponses(prev => ({
                ...prev,
                [selectedSnippet]: JSON.stringify(capturedLogs, null, 2)
            }));
        } catch (error) {
            console.error('Error:', error);
            setErrorMessage(error.message || 'Failed to execute code');
        } finally {
            setIsLoading(false);
        }
    };

    const getConsoleOutput = () => {
        const currentResponse = apiResponses[selectedSnippet];
        if (!currentResponse) return null;

        try {
            const logs = JSON.parse(currentResponse);
            if (!logs || logs.length === 0) return 'No output captured';

            // Check if any logs contain image URLs (specific format: description, imageUrl)
            const processedLogs = logs.map(log => {
                // Check if log matches the pattern "description: url" where url looks like an image URL
                // Updated regex to handle Poe CDN URLs and traditional image extensions
                const imageMatch = log.match(/^(.+?):\s*(https?:\/\/[^\s]+(?:\.(jpg|jpeg|png|gif|webp|svg)(?:\?[^\s]*)?|\/image\/[^\s?]+(?:\?[^\s]*)?|pfst\.cf2\.poecdn\.net[^\s]+))/i);
                if (imageMatch) {
                    const description = imageMatch[1];
                    const imageUrl = imageMatch[2];
                    return `${description}:\n<img src="${imageUrl}" alt="${description}" style="max-width: 400px; max-height: 400px; border-radius: 8px; margin: 10px 0;" />`;
                }
                return log;
            });

            return processedLogs.join('\n');
        } catch (e) {
            return "Could not parse console output";
        }
    };

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Code copied to clipboard!');
            })
            .catch(err => {
                console.error('Failed to copy: ', err);
            });
    };


    return (
        <div className="playground-section">
            <div className="playground-container">
                <button
                    className="playground-toggle"
                    onClick={() => setIsCollapsed(!isCollapsed)}
                >
                    <span className="playground-title"><Beaker size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} />API Playground</span>
                    <span className="toggle-icon">{isCollapsed ? '▼' : '▲'}</span>
                </button>

                {!isCollapsed && (
                    <div className="playground-content">
                        <p>Test API requests directly in your browser.</p>

                        <div className="api-key-input">
                            <label htmlFor="api-key">API Key:</label>
                            <input
                                type="text"
                                id="api-key"
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                placeholder="Enter your API key from https://poe.com/api_key"
                                onKeyDown={(e) => {
                                    // Prevent the Enter key from triggering any form-like submission
                                    if (e.key === 'Enter') {
                                        e.preventDefault();
                                    }
                                }}
                                autoComplete="off"
                                data-lpignore="true"
                                style={{ fontFamily: 'monospace', WebkitTextSecurity: 'disc' }}
                            />
                        </div>

                        <div className="snippet-selector">
                            <h4>Code Snippets</h4>
                            <Tabs tabs={snippetTabs} activeTab={selectedSnippet} onChange={setSelectedSnippet} />
                        </div>

                        <div className="snippet-content">
                            <h4>{codeSnippets[selectedSnippet].name}</h4>
                            <p>{codeSnippets[selectedSnippet].description}</p>
                            <CodeBlock
                                code={codeSnippets[selectedSnippet].code}
                                language="javascript"
                                onCopy={copyToClipboard}
                            />

                            {apiResponses[selectedSnippet] && (
                                <div className="response-container">
                                    <div className="response-content">
                                        <h4>Console Output:</h4>
                                        <div className="content-box">
                                            <div
                                                style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'monospace' }}
                                                dangerouslySetInnerHTML={{ __html: getConsoleOutput() }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}

                            <button
                                className="run-button"
                                onClick={executeCode}
                                disabled={isLoading || !apiKey}
                                title={!apiKey ? "Token is required" : ""}
                            >
                                {isLoading ? 'Executing...' : 'Run snippet'}
                            </button>
                        </div>

                        {errorMessage && (
                            <div className="error-message">
                                <strong>Error:</strong> {errorMessage}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function ApiQuickstart() {
    const [exampleTab, setExampleTab] = useState('javascript')
    const [models, setModels] = useState([])
    const [modelsLoading, setModelsLoading] = useState(false)
    const [modelsError, setModelsError] = useState('')

    const exampleTabs = [
        { id: 'javascript', label: 'JavaScript' },
        { id: 'python', label: 'Python' },
        { id: 'cline', label: 'Cline/Roo' },
        { id: 'curl', label: 'cURL' }
    ]

    // Function to copy code to clipboard
    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Code copied to clipboard!');
            })
            .catch(err => {
                console.error('Failed to copy: ', err)
            })
    }

    // Utility function to format token count nicely
    const formatTokenCount = (count) => {
        // If count is undefined or null, return a default value
        if (count === undefined || count === null) return "100k";

        if (count >= 1000000) {
            return `${(count / 1000000).toFixed(1).replace(/\.0$/, '')}M`;
        } else if (count >= 1000) {
            return `${(count / 1000).toFixed(1).replace(/\.0$/, '')}k`;
        } else {
            return count.toString();
        }
    }

    // Function to fetch available models from the API
    const fetchModels = async (apiKey) => {
        setModelsLoading(true)
        setModelsError('')

        try {
            // For demonstration, we'll use a mock API key if none is provided
            const demoApiKey = apiKey || 'demo-key'

            const response = await fetch(`${API_BASE_URL}/models`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${demoApiKey}`
                }
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.error?.message || 'Error fetching models')
            }

            // Extract model information from the response
            // The OpenAI-compatible API usually returns models in data array
            if (data && data.data) {

                // Process models to ensure token lengths
                const processedModels = data.data.map(model => {
                    // Assign a default token length based on model name if not provided
                    let tokenLength;
                    if (model.id.toLowerCase().includes('claude')) {
                        tokenLength = 200000;
                    } else if (model.id.toLowerCase().includes('gpt')) {
                        tokenLength = 128000;
                    } else if (model.id.toLowerCase().includes('gemini')) {
                        tokenLength = 1000000;
                    } else {
                        tokenLength = 100000; // Default fallback
                    }

                    return {
                        ...model,
                        // Use the API's token count if available, otherwise use our estimated value
                        tokenCount: model.token_limit || model.tokenLimit || model.maxTokens || model.contextLength || tokenLength
                    };
                });

                setModels(processedModels);
            } else {
                setModels([]);
                console.warn('Models data format unexpected:', data);
            }
        } catch (error) {
            console.error('Error fetching models:', error)
            setModelsError(error.message || 'Failed to fetch models')
            // If we can't fetch models, set default ones so UI still works
            setModels([])
        } finally {
            setModelsLoading(false)
        }
    }

    // Fetch models when component mounts
    useEffect(() => {
        // In a real app, you might want to get the API key from a state or context
        // Here we just fetch without a key to show the error handling flow
        fetchModels()
    }, [])

    // Get the default model - first available model or fallback
    const defaultModel = models.length > 0 ? models[0].id : 'Claude-3.7-Sonnet'

    const javascriptCode = `
// First, install the OpenAI SDK
// npm install openai

import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: "YOUR_POE_API_KEY", // Get your API key from https://poe.com/api_key
  baseURL: "${API_BASE_URL}",
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    model: '${defaultModel}',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello, how are you today?' }
    ],
  });

  console.log(chatCompletion.choices[0].message.content);
}

main();
`

    const pythonCode = `
# First, install the OpenAI SDK
# pip install openai

from openai import OpenAI

client = OpenAI(
    api_key="YOUR_POE_API_KEY",  # Get your API key from https://poe.com/api_key
    base_url="${API_BASE_URL}"
)

chat_completion = client.chat.completions.create(
    model="${defaultModel}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"}
    ]
)

print(chat_completion.choices[0].message.content)
`

    return (
        <div className="api-docs">
            <h1>Unofficial Poe API Bridge Documentation</h1>

            <p>
                This API bridge provides OpenAI-compatible access to Poe's bot ecosystem.
                Use your existing OpenAI SDK with minimal configuration changes to access
                Claude, GPT-4, and other advanced models through a unified interface.
            </p>

            <div className="quick-start-section">
                <h2><Rocket size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} />Quick Start</h2>
                <div className="setup-steps">
                    <div className="step">
                        <div className="step-number">1</div>
                        <div className="step-content">
                            <p>Create an account or login to <a href="https://poe.com" target="_blank" rel="noopener noreferrer">Poe</a></p>
                        </div>
                    </div>
                    <div className="step">
                        <div className="step-number">2</div>
                        <div className="step-content">
                            <p>Go to <a href="https://poe.com/api_key" target="_blank" rel="noopener noreferrer">API Keys</a> and get your API key.</p>
                        </div>
                    </div>
                    <div className="step">
                        <div className="step-number">3</div>
                        <div className="step-content">
                            <p>Use any OpenAI-compatible SDK with the following configuration:</p>
                            <div className="config-box">
                                <div className="config-item">
                                    <span className="config-label">Base URL:</span>
                                    <InlineCode copyable={true}>{API_BASE_URL}</InlineCode>
                                </div>
                                <div className="config-item">
                                    <span className="config-label">API Key:</span>
                                    <InlineCode copyable={true}>YOUR_POE_API_KEY</InlineCode>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="api-docs-note">
                <p><strong>Current Limitations:</strong></p>
                <ul className="simple-list">
                    <li>Prompt caching functionality is under development on Poe side.</li>
                    <li>Tool calling is done via prompting/parsing. Poe doesn't support native Tool calling.</li>
                </ul>
                <p className="note-footer">
                    Track development progress on <a href="https://github.com/poe-platform/fastapi_poe" target="_blank" rel="noopener noreferrer">GitHub</a>.
                </p>
            </div>



            <h2><Wrench size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} />Integration Examples</h2>
            <p>Select your preferred language or tool to view integration examples:</p>

            <Tabs tabs={exampleTabs} activeTab={exampleTab} onChange={setExampleTab} />

            <div className="tab-content">
                {exampleTab === 'javascript' && (
                    <div>
                        <h3>JavaScript</h3>
                        <p>Example chat completion request using the OpenAI SDK:</p>
                        <CodeBlock code={javascriptCode} language="javascript" onCopy={copyToClipboard} />
                    </div>
                )}

                {exampleTab === 'python' && (
                    <div>
                        <h3>Python</h3>
                        <p>Example chat completion request using the OpenAI Python library:</p>
                        <CodeBlock code={pythonCode} language="python" onCopy={copyToClipboard} />
                    </div>
                )}

                {exampleTab === 'cline' && (
                    <div>
                        <h3>Cline/Roo Configuration</h3>
                        <p>Configure Cline (Roo) VS Code extension to use this API bridge:</p>
                        <ol>
                            <li>
                                <strong>Install Cline Extension</strong>
                                <p>Available at <a href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev" target="_blank" rel="noopener noreferrer">VS Code Marketplace</a></p>
                            </li>

                            <li>
                                <strong>Open Configuration</strong>
                                <p>Press <InlineCode>Ctrl/Cmd+Shift+P</InlineCode> and select "Cline: Open Settings"</p>
                            </li>

                            <li>
                                <strong>Configure API Settings</strong>
                                <ul>
                                    <li>API Provider: <InlineCode>OpenAI Compatible</InlineCode></li>
                                    <li>Base URL: <InlineCode copyable>{API_BASE_URL}</InlineCode></li>
                                    <li>API Key: <InlineCode copyable>YOUR_POE_API_KEY</InlineCode></li>
                                    <li>Model ID: <InlineCode copyable>{defaultModel}</InlineCode></li>
                                </ul>
                            </li>
                        </ol>
                    </div>
                )}

                {exampleTab === 'curl' && (
                    <div>
                        <h3>cURL</h3>
                        <p>Direct API access using cURL:</p>

                        <h4>Chat Completion Example</h4>
                        <CodeBlock
                            code={`curl ${API_BASE_URL}/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer YOUR_POE_API_KEY" \\
    -d '{
      "model": "${defaultModel}",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"}
      ]
    }'`}
                            language="bash"
                            onCopy={copyToClipboard}
                        />
                    </div>
                )}
            </div>

            <h2><Bot size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} />Available Models</h2>
            <p>Access all Poe bots through this API bridge including these. Check out <a href="https://poe.com/explore?category=Official" target="_blank" rel="noopener noreferrer">all official bots</a> and <a href="https://poe.com/explore?category=Image+generation" target="_blank" rel="noopener noreferrer">image generation bots</a>.</p>

            {modelsLoading && <p>Loading available models...</p>}

            {modelsError && (
                <div className="error-message">
                    <p>Error loading models: {modelsError}</p>
                    <p>Showing recommended models instead:</p>
                </div>
            )}

            <table>
                <thead>
                    <tr>
                        <th>Model ID</th>
                        <th>Modality</th>
                        <th>Token Length</th>
                    </tr>
                </thead>
                <tbody>
                    {models && models.length > 0 && models.map((model, index) => (
                        <tr key={index}>
                            <td><InlineCode copyable={true}>{model.id}</InlineCode></td>
                            <td>{model.capabilities?.multimodal ? "text→text, image→text" : "text→text"}</td>
                            <td>{formatTokenCount(model.tokenCount)}</td>
                        </tr>
                    ))}
                    {(!models || models.length === 0) && (
                        <tr>
                            <td colSpan="3" style={{ textAlign: "center" }}>No models available</td>
                        </tr>
                    )}
                </tbody>
            </table>


            <div className="info-box" style={{ marginTop: "2rem" }}>
                <strong><HelpCircle size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} />Support</strong>
                <p>For questions, bug reports, or feature requests: <a href="https://docs.google.com/forms/d/e/1FAIpQLScTOeInU9c0gCC3yolhPkU05TPbZTf68jcDGECIm8nq0u9Yrg/viewform?usp=header" target="_blank" rel="noopener noreferrer">Submit feedback</a></p>
            </div>

            {/* Add the API Playground component */}
            <ApiPlayground defaultModel={defaultModel} />
        </div>
    )
}

export default ApiQuickstart