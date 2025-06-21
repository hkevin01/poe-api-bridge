import React, { useState, useRef, useEffect } from 'react';
import './Tabs.css';

const Tabs = ({ tabs, activeTab, onChange }) => {
    const [tabsRect, setTabsRect] = useState({});
    const tabRefs = useRef(new Map());
    const containerRef = useRef(null);
    const indicatorRef = useRef(null);

    useEffect(() => {
        const updateTabPositions = () => {
            const newRects = {};
            tabRefs.current.forEach((ref, key) => {
                if (ref) {
                    const rect = ref.getBoundingClientRect();
                    const containerRect = containerRef.current.getBoundingClientRect();
                    newRects[key] = {
                        left: rect.left - containerRect.left,
                        width: rect.width,
                    };
                }
            });
            setTabsRect(newRects);
        };

        updateTabPositions();
        window.addEventListener('resize', updateTabPositions);
        return () => window.removeEventListener('resize', updateTabPositions);
    }, [tabs]);

    useEffect(() => {
        if (indicatorRef.current && tabsRect[activeTab]) {
            indicatorRef.current.style.left = `${tabsRect[activeTab].left}px`;
            indicatorRef.current.style.width = `${tabsRect[activeTab].width}px`;
        }
    }, [tabsRect, activeTab]);

    const handleKeyDown = (e, tab) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onChange(tab.id);
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            const currentIndex = tabs.findIndex(t => t.id === activeTab);
            if (currentIndex < tabs.length - 1) {
                onChange(tabs[currentIndex + 1].id);
            }
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const currentIndex = tabs.findIndex(t => t.id === activeTab);
            if (currentIndex > 0) {
                onChange(tabs[currentIndex - 1].id);
            }
        }
    };

    return (
        <div
            className="relative"
            ref={containerRef}
            role="tablist"
            aria-orientation="horizontal"
        >
            <div className="flex space-x-1 border-b border-gray-200 relative">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        ref={(el) => tabRefs.current.set(tab.id, el)}
                        role="tab"
                        aria-selected={activeTab === tab.id}
                        aria-controls={`panel-${tab.id}`}
                        id={`tab-${tab.id}`}
                        tabIndex={activeTab === tab.id ? 0 : -1}
                        className={`
              relative px-4 py-2 text-sm font-medium rounded-t-lg
              transition-colors duration-200 ease-in-out
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              ${activeTab === tab.id
                                ? 'text-blue-600 bg-white'
                                : 'text-gray-600 hover:text-gray-700 hover:bg-gray-50'
                            }
            `}
                        onClick={() => onChange(tab.id)}
                        onKeyDown={(e) => handleKeyDown(e, tab)}
                    >
                        {tab.label}
                    </button>
                ))}
                <div
                    ref={indicatorRef}
                    className="absolute bottom-0 h-0.5 bg-blue-500 transition-all duration-200 ease-in-out"
                    style={{ left: 0, width: 0 }}
                />
            </div>
        </div>
    );
};

export default Tabs;