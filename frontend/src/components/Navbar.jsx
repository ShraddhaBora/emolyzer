// components/Navbar.jsx
import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const TABS = [
    { id: 'predict', label: 'Analyse' },
    { id: 'distribution', label: 'Dataset' },
    { id: 'metrics', label: 'Performance' },
]

// Animated Emolyzer icon — lines draw in on mount, gently rocks on idle
function EmolyzerIcon() {
    return (
        <motion.svg
            width="20" height="20" viewBox="0 0 24 24"
            fill="none" aria-hidden="true"
            stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
            animate={{ rotate: [0, -5, 5, -3, 0] }}
            transition={{ duration: 5, repeat: Infinity, repeatDelay: 4, ease: 'easeInOut' }}
        >
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <motion.path
                d="M7 9h10"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 0.65, delay: 0.1, ease: 'easeOut' }}
            />
            <motion.path
                d="M7 14h6"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 0.45, delay: 0.4, ease: 'easeOut' }}
            />
        </motion.svg>
    )
}

export default function Navbar({ activeTab, onTabChange }) {
    const [scrolled, setScrolled] = useState(false)
    const [hovered, setHovered] = useState(null)

    useEffect(() => {
        const handler = () => setScrolled(window.scrollY > 12)
        window.addEventListener('scroll', handler, { passive: true })
        return () => window.removeEventListener('scroll', handler)
    }, [])

    return (
        <motion.nav
            className="navbar"
            animate={{ boxShadow: scrolled ? '0 4px 24px rgba(42,54,68,0.08)' : '0 0px 0px rgba(0,0,0,0)' }}
            transition={{ duration: 0.3 }}
        >
            <div className="navbar-inner">

                {/* ── Animated Logo ── */}
                <motion.a
                    href="/"
                    className="logo"
                    onClick={e => e.preventDefault()}
                    whileHover="hover"
                    initial="rest"
                    animate="rest"
                >
                    <motion.span
                        className="logo-icon"
                        variants={{
                            rest: { scale: 1, rotate: 0 },
                            hover: {
                                scale: 1.18,
                                rotate: -10,
                                transition: { type: 'spring', stiffness: 450, damping: 12 }
                            }
                        }}
                    >
                        <EmolyzerIcon />
                    </motion.span>
                    <motion.span
                        style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}
                        variants={{
                            rest: { letterSpacing: '0.05em', opacity: 1 },
                            hover: {
                                letterSpacing: '0.13em',
                                opacity: 0.8,
                                transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] }
                            }
                        }}
                    >
                        EMOLYZER
                    </motion.span>
                </motion.a>

                {/* ── Tabs with shared spring pill ── */}
                <div className="nav-tabs" style={{ display: 'flex', position: 'relative' }}>
                    {TABS.map((tab) => {
                        const isActive = activeTab === tab.id
                        const isHovered = hovered === tab.id

                        return (
                            <motion.button
                                key={tab.id}
                                className="nav-tab"
                                onClick={() => onTabChange(tab.id)}
                                onHoverStart={() => setHovered(tab.id)}
                                onHoverEnd={() => setHovered(null)}
                                whileTap={{ scale: 0.84, transition: { type: 'spring', stiffness: 600, damping: 18 } }}
                                style={{ position: 'relative', border: 'none', background: 'transparent' }}
                            >
                                {/* Hover shimmer bg */}
                                <AnimatePresence>
                                    {isHovered && !isActive && (
                                        <motion.span
                                            style={{
                                                position: 'absolute', inset: 0,
                                                background: 'rgba(42,54,68,0.05)',
                                                borderRadius: 'var(--r-sm)',
                                                zIndex: 0,
                                            }}
                                            initial={{ opacity: 0, scale: 0.92 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.96 }}
                                            transition={{ duration: 0.16 }}
                                        />
                                    )}
                                </AnimatePresence>

                                {/* Sliding active pill — shared layoutId makes it animate between tabs */}
                                {isActive && (
                                    <motion.span
                                        layoutId="active-pill"
                                        style={{
                                            position: 'absolute', inset: 0,
                                            background: 'var(--surface)',
                                            border: '1px solid var(--border)',
                                            borderRadius: 'var(--r-sm)',
                                            boxShadow: '0 2px 10px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04)',
                                            zIndex: 0,
                                        }}
                                        transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                                    />
                                )}

                                {/* Accent underline on active tab */}
                                {isActive && (
                                    <motion.span
                                        layoutId="active-line"
                                        style={{
                                            position: 'absolute',
                                            bottom: 1, left: '22%', right: '22%',
                                            height: '2px',
                                            background: 'var(--accent)',
                                            borderRadius: '2px 2px 0 0',
                                            zIndex: 1,
                                        }}
                                        transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                                    />
                                )}

                                {/* Label — color transitions smoothly */}
                                <motion.span
                                    style={{ position: 'relative', zIndex: 1, fontSize: '0.85rem', fontFamily: 'var(--font-mono)', fontWeight: 500 }}
                                    animate={{
                                        color: isActive
                                            ? 'var(--text-primary)'
                                            : isHovered
                                                ? 'var(--text-secondary)'
                                                : 'var(--text-muted)',
                                        y: isHovered && !isActive ? -1 : 0,
                                    }}
                                    transition={{ duration: 0.18, ease: 'easeOut' }}
                                >
                                    {tab.label}
                                </motion.span>
                            </motion.button>
                        )
                    })}
                </div>
            </div>
        </motion.nav>
    )
}
