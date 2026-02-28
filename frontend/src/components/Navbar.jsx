// components/Navbar.jsx
import React from 'react'

const TABS = [
    { id: 'predict', label: 'Analyse' },
    { id: 'distribution', label: 'Dataset' },
    { id: 'metrics', label: 'Performance' },
]

export default function Navbar({ activeTab, onTabChange }) {
    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <a href="/" className="logo">
                    <span className="logo-icon">Em</span>
                    Emolyzer
                </a>
                <div className="nav-tabs">
                    {TABS.map((tab) => (
                        <button
                            key={tab.id}
                            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                            onClick={() => onTabChange(tab.id)}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>
        </nav>
    )
}
