import React, { useState } from 'react';
import { Highlight, themes } from 'prism-react-renderer';

const CodeBlock = ({ code, language, onCopy }) => {
    const [isCopied, setIsCopied] = useState(false);

    const handleCopy = () => {
        onCopy(code.trim());
        setIsCopied(true);
        setTimeout(() => {
            setIsCopied(false);
        }, 2000); // Reset after 2 seconds
    };

    return (
        <Highlight
            theme={themes.vsDark}
            code={code.trim()}
            language={language}
        >
            {({ className, style, tokens, getLineProps, getTokenProps }) => (
                <pre className={`${className} relative`} style={{ ...style, padding: '1rem', borderRadius: '0.5rem', margin: '1rem 0', overflowX: 'auto' }}>
                    {onCopy && (
                        <button
                            className="copy-button"
                            style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', backgroundColor: isCopied ? '#38a169' : '#4a5568', color: 'white', border: 'none', borderRadius: '0.25rem', padding: '0.25rem 0.5rem', fontSize: '0.75rem', cursor: 'pointer', opacity: 0.8, transition: 'opacity 0.2s, background-color 0.2s' }}
                            onMouseOver={(e) => e.currentTarget.style.opacity = '1'}
                            onMouseOut={(e) => e.currentTarget.style.opacity = '0.8'}
                            onClick={handleCopy}
                            disabled={isCopied}
                        >
                            {isCopied ? 'Copied!' : 'Copy'}
                        </button>
                    )}
                    {tokens.map((line, i) => (
                        <div key={i} {...getLineProps({ line })}>
                            {line.map((token, key) => (
                                <span key={key} {...getTokenProps({ token })} />
                            ))}
                        </div>
                    ))}
                </pre>
            )}
        </Highlight>
    );
};

export default CodeBlock;