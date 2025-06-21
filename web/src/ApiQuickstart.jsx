import React, { useState, useEffect } from 'react'
import Tabs from './components/Tabs'
import CodeBlock from './components/CodeBlock'
import InlineCode from './components/InlineCode'
import { Beaker, Rocket, Wrench, Lock, Bot, HelpCircle } from 'lucide-react'
import './ApiQuickstart.css'

// API base URL constant
const API_BASE_URL = 'https://kamilio--poe-api-bridge-poeapibridge-fastapi-app.modal.run/v1'

// ApiPlayground Component
function ApiPlayground({ defaultModel }) {
    const [isCollapsed, setIsCollapsed] = useState(true)
    const [apiKey, setApiKey] = useState('')
    const [apiResponse, setApiResponse] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')

    const vanillaJsCode = `// Vanilla JavaScript example
const apiKey = "${apiKey || 'YOUR_POE_API_KEY'}"; // Your API key from https://poe.com/api_key
const baseUrl = "${API_BASE_URL}";

// Function to make a request to the Poe API
async function makeRequest() {
  try {
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
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}`

    const handleRunExample = async () => {
        if (!apiKey) {
            setErrorMessage('Please enter your API key.');
            return;
        }

        setIsLoading(true);
        setApiResponse('');
        setErrorMessage('');

        try {
            const response = await fetch(`${API_BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: defaultModel,
                    messages: [
                        { role: "system", content: "You are a helpful assistant." },
                        { role: "user", content: "Tell me a short joke about programming." }
                    ]
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error?.message || 'Unknown error occurred');
            }

            setApiResponse(JSON.stringify(data, null, 2));
        } catch (error) {
            console.error('Error:', error);
            setErrorMessage(error.message || 'Failed to fetch response');
        } finally {
            setIsLoading(false);
        }
    };

    const extractContent = (responseJson) => {
        try {
            const data = JSON.parse(responseJson);
            return data.choices[0].message.content;
        } catch (e) {
            return "Could not parse response content";
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

    const [jsonCollapsed, setJsonCollapsed] = useState(true);

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

                        <h4>JavaScript Example</h4>
                        <p>This code will make a simple chat completion request to the Poe API:</p>
                        <CodeBlock
                            code={vanillaJsCode}
                            language="javascript"
                            onCopy={copyToClipboard}
                        />

                        <button
                            className="run-button"
                            onClick={handleRunExample}
                            disabled={isLoading}
                        >
                            {isLoading ? 'Loading...' : 'Run Example'}
                        </button>

                        {errorMessage && (
                            <div className="error-message">
                                <strong>Error:</strong> {errorMessage}
                            </div>
                        )}

                        {apiResponse && (
                            <div className="response-container">
                                {apiResponse.includes('"content":') && (
                                    <div className="response-content">
                                        <h4>Response Content:</h4>
                                        <div className="content-box">
                                            {extractContent(apiResponse)}
                                        </div>
                                    </div>
                                )}

                                <div className="json-response">
                                    <button
                                        className="json-toggle"
                                        onClick={() => setJsonCollapsed(!jsonCollapsed)}
                                    >
                                        <h4>API Response (JSON):</h4>
                                        <span>{jsonCollapsed ? '▼' : '▲'}</span>
                                    </button>

                                    {!jsonCollapsed && (
                                        <CodeBlock
                                            code={apiResponse}
                                            language="json"
                                            onCopy={copyToClipboard}
                                        />
                                    )}
                                </div>
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
                            <p>Subscribe to <a href="https://poe.com" target="_blank" rel="noopener noreferrer">Poe</a> or sign in if you already have an account.</p>
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
                    <li>Media attachments (images, video, audio) are not yet fully supported. Some models will return content URL. Try it out.</li>
                    <li>Prompt caching functionality is under development</li>
                    <li>Tool/function calling is not available. You can simulate tool calling via prompting. XML works best.</li>
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

                        <h4>List Available Models</h4>
                        <CodeBlock
                            code={`
// First, install the OpenAI SDK
// npm install openai

import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: "YOUR_POE_API_KEY", // Get your API key from https://poe.com/api_key
  baseURL: "${API_BASE_URL}",
});

async function listModels() {
  const models = await openai.models.list();
  console.log(models.data);
}

listModels();`}
                            language="javascript"
                            onCopy={copyToClipboard}
                        />
                    </div>
                )}

                {exampleTab === 'python' && (
                    <div>
                        <h3>Python</h3>
                        <p>Example chat completion request using the OpenAI Python library:</p>
                        <CodeBlock code={pythonCode} language="python" onCopy={copyToClipboard} />

                        <h4>List Available Models</h4>
                        <CodeBlock
                            code={`
# First, install the OpenAI SDK
# pip install openai

from openai import OpenAI

client = OpenAI(
    api_key="YOUR_POE_API_KEY",  # Get your API key from https://poe.com/api_key
    base_url="${API_BASE_URL}"
)

# List available models
models = client.models.list()
for model in models.data:
    print(model.id)`}
                            language="python"
                            onCopy={copyToClipboard}
                        />
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

                        <h4>List Available Models Example</h4>
                        <CodeBlock
                            code={`curl ${API_BASE_URL}/models \\
    -H "Authorization: Bearer YOUR_POE_API_KEY"`}
                            language="bash"
                            onCopy={copyToClipboard}
                        />
                    </div>
                )}
            </div>

            <h2><Bot size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} />Available Models</h2>
            <p>Access all Poe bots through this API bridge including these</p>

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