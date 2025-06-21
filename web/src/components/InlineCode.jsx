import React, { useState } from 'react';
import './InlineCode.css'; // We'll create this for styling

const InlineCode = ({ children, copyable = false }) => {
    const [isCopied, setIsCopied] = useState(false);

    // Function to copy text to clipboard
    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Text copied to clipboard!');
                setIsCopied(true);
                setTimeout(() => {
                    setIsCopied(false);
                }, 2000); // Reset after 2 seconds
            })
            .catch(err => {
                console.error('Failed to copy: ', err);
            });
    };

    return (
        <span className="inline-code-container">
            <code className="inline-code">
                {children}
                {copyable && (
                    <button
                        onClick={() => copyToClipboard(children)}
                        className={`copy-button-inline ${isCopied ? 'copied' : ''}`}
                        title="Copy to clipboard"
                        disabled={isCopied}
                    >
                        {isCopied ? 'Copied!' : 'Copy'}
                    </button>
                )}
            </code>
        </span>
    );
};

export default InlineCode;