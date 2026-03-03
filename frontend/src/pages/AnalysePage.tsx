// @ts-nocheck
// pages/AnalysePage.jsx
import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiPredict } from '../api'
import ProbabilityBars from '../components/ProbabilityBars'

const EMOTION_EMOJI = {
    Sadness: '😢',
    Joy: '😄',
    Love: '❤️',
    Anger: '😠',
    Fear: '😨',
    Surprise: '😲',
    Neutral: '😐',
    'Unknown (Out of Vocabulary)': '🤷',
}

const EXAMPLE_INPUTS = [
    "I can't believe how wonderful today has been!",
    'This is absolutely infuriating, I hate it.',
    'I miss my family so much it hurts.',
    'Oh wow, I did not see that coming at all.',
]

// Clean slow opacity fade — research-grade, no movement, no blur
const fadeUp = (delay = 0) => ({
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: 0.6, delay, ease: 'easeOut' },
})

// Clean inline SVG icons
function IconScan() {
    return (
        <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ flexShrink: 0 }}
        >
            <path d="M3 7V5a2 2 0 0 1 2-2h2" />
            <path d="M17 3h2a2 2 0 0 1 2 2v2" />
            <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
            <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
            <circle cx="12" cy="12" r="3" />
            <path d="M12 9v-2M12 17v-2M9 12H7M17 12h-2" />
        </svg>
    )
}

function IconBrainPulse() {
    return (
        <svg
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
        >
            <path d="M9.5 2a2.5 2.5 0 0 1 2.45 2H13a3 3 0 0 1 3 3c0 .35-.07.69-.18 1" />
            <path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v1A3.5 3.5 0 0 0 3.5 9c0 1.5.94 2.8 2.3 3.3" />
            <path d="M16 7a3 3 0 0 1 2.83 4" />
            <path d="M18.83 11A3.5 3.5 0 0 1 20.5 14 3.5 3.5 0 0 1 17 17.5h-.5" />
            <path d="M5.8 12.3A4 4 0 0 0 5 14.5 3.5 3.5 0 0 0 8.5 18H10" />
            <path d="M10 22v-9" />
            <path d="M14 22v-9" />
            <path d="M7 17l3-3 2 2 3-3 2 2" />
        </svg>
    )
}

// Animated circular confidence arc
function ConfidenceArc({ value, color }: { value: number; color: string }) {
    const r = 28
    const circ = 2 * Math.PI * r
    const fill = circ * (1 - value)
    return (
        <div className="confidence-arc-wrap">
            <svg width="72" height="72" className="confidence-arc-svg">
                <circle
                    cx="36"
                    cy="36"
                    r={r}
                    fill="none"
                    stroke="rgba(196,186,255,0.15)"
                    strokeWidth="5"
                />
                <motion.circle
                    cx="36"
                    cy="36"
                    r={r}
                    fill="none"
                    stroke={color}
                    strokeWidth="5"
                    strokeLinecap="round"
                    strokeDasharray={circ}
                    initial={{ strokeDashoffset: circ }}
                    animate={{ strokeDashoffset: fill }}
                    transition={{
                        duration: 1.1,
                        ease: [0.16, 1, 0.3, 1],
                        delay: 0.1,
                    }}
                />
            </svg>
            <div
                style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 700,
                    fontSize: '0.85rem',
                    color,
                    fontFamily: 'var(--font-mono)',
                }}
            >
                {(value * 100).toFixed(0)}%
            </div>
        </div>
    )
}

export default function AnalysePage({ apiReady }: { apiReady: boolean }) {
    const [text, setText] = useState('')
    const [result, setResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [charCount, setCharCount] = useState(0)

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        if (!text.trim()) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const data = await apiPredict(text)
            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    function handleExample(ex: string) {
        setText(ex)
        setCharCount(ex.length)
        setResult(null)
        setError(null)
    }

    function handleTextChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
        setText(e.target.value)
        setCharCount(e.target.value.length)
    }

    const color =
        result?.probabilities?.[result.predicted_emotion]?.color ??
        'var(--accent)'

    return (
        <div
            className="container"
            style={{ paddingTop: '2rem', paddingBottom: '4rem' }}
        >
            {/* ── Hero — each element staggered by 200ms, NO overlap ── */}
            <section className="hero" style={{ paddingTop: 0 }}>
                <motion.h1 {...fadeUp(0.05)}>
                    Understand the{' '}
                    <span className="gradient-text">emotion</span> behind any
                    text
                </motion.h1>
                <motion.p {...fadeUp(0.22)}>
                    Enter any sentence, message, or phrase. Our multi-model
                    pipeline classifies it into one of seven core emotions with
                    calibrated confidence scores.
                </motion.p>
            </section>

            {/* ── Example chips (now staggered individually) ── */}
            <div
                style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    justifyContent: 'center',
                    marginBottom: '2rem',
                }}
            >
                {EXAMPLE_INPUTS.map((ex, i) => (
                    <motion.button
                        key={ex}
                        className="chip"
                        onClick={() => handleExample(ex)}
                        {...fadeUp(0.4 + i * 0.08)}
                    >
                        {ex.length > 40 ? ex.slice(0, 40) + '…' : ex}
                    </motion.button>
                ))}
            </div>

            {/* ── Main predictor grid ── */}
            <div className="predictor-wrap">
                {/* Input card — delay 0.55 so chips are ~done */}
                <motion.form
                    onSubmit={handleSubmit}
                    className="card"
                    {...fadeUp(0.55)}
                >
                    <div
                        style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '0.75rem',
                        }}
                    >
                        <p className="section-title" style={{ margin: 0 }}>
                            Your Text
                        </p>
                        <span
                            style={{
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                                fontFamily: 'var(--font-mono)',
                            }}
                        >
                            {charCount} chars
                        </span>
                    </div>

                    <textarea
                        id="text-input"
                        className="predictor-textarea"
                        rows={5}
                        placeholder="Type or paste any text here…"
                        value={text}
                        onChange={handleTextChange}
                    />

                    {!apiReady && (
                        <p
                            style={{
                                fontSize: '0.8rem',
                                color: 'var(--text-muted)',
                                marginTop: '0.5rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.4rem',
                            }}
                        >
                            <span
                                className="spinner-dots"
                                style={{ transform: 'scale(0.7)' }}
                            >
                                <span />
                                <span />
                                <span />
                            </span>
                            Model is still training, please wait…
                        </p>
                    )}

                    <button
                        id="analyse-btn"
                        type="submit"
                        className="btn btn-primary"
                        disabled={!text.trim() || loading || !apiReady}
                    >
                        {loading ? (
                            <>
                                <span
                                    style={{
                                        width: 15,
                                        height: 15,
                                        border: '2px solid rgba(255,255,255,0.35)',
                                        borderTopColor: '#fff',
                                        borderRadius: '50%',
                                        display: 'inline-block',
                                        animation: 'spin 0.7s linear infinite',
                                        flexShrink: 0,
                                    }}
                                />
                                Analysing…
                            </>
                        ) : (
                            <>
                                <IconScan />
                                Analyse Emotion
                            </>
                        )}
                    </button>

                    {error && (
                        <div
                            className="oov-notice oov-notice-error"
                            style={{ marginTop: '0.75rem' }}
                        >
                            ⚠ {error}
                        </div>
                    )}
                </motion.form>

                {/* Result panel — same y-axis, delay 0.65 so form is ~done */}
                <motion.div {...fadeUp(0.65)}>
                    <AnimatePresence mode="wait">
                        {/* Placeholder — simple element wave */}
                        {!result && !loading && (
                            <motion.div
                                key="placeholder"
                                className="result-badge"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                style={{ gap: '1.2rem' }}
                            >
                                <div
                                    className="wave-loader wave-idle"
                                    style={{ height: 20 }}
                                >
                                    {[0, 1, 2, 3, 4].map((i) => (
                                        <div
                                            key={i}
                                            className="wave-bar"
                                            style={{
                                                background: 'var(--text-muted)',
                                                opacity: 0.25,
                                                width: 3,
                                                margin: '0 2px',
                                            }}
                                        />
                                    ))}
                                </div>
                                <p
                                    style={{
                                        color: 'var(--text-muted)',
                                        fontSize: '0.82rem',
                                        fontFamily: 'var(--font-mono)',
                                        letterSpacing: '0.05em',
                                    }}
                                >
                                    awaiting input
                                </p>
                            </motion.div>
                        )}

                        {/* Wave loader while analysing */}
                        {loading && (
                            <motion.div
                                key="loading"
                                className="result-badge"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '0.75rem',
                                }}
                            >
                                <div className="wave-loader">
                                    {[
                                        'var(--em-sadness)',
                                        'var(--em-joy)',
                                        'var(--em-love)',
                                        'var(--em-anger)',
                                        'var(--em-fear)',
                                        'var(--em-surprise)',
                                        'var(--em-neutral)',
                                    ].map((c, i) => (
                                        <div
                                            key={i}
                                            className="wave-bar"
                                            style={{ background: c }}
                                        />
                                    ))}
                                </div>
                                <p
                                    style={{
                                        color: 'var(--text-muted)',
                                        fontSize: '0.85rem',
                                        fontFamily: 'var(--font-mono)',
                                    }}
                                >
                                    Analysing emotion…
                                </p>
                            </motion.div>
                        )}

                        {/* Result */}
                        {result && !loading && (
                            <motion.div
                                key="result"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                transition={{
                                    opacity: {
                                        duration: 0.45,
                                        ease: 'easeOut',
                                    },
                                    y: {
                                        duration: 0.5,
                                        ease: [0.16, 1, 0.3, 1],
                                    },
                                }}
                            >
                                <div
                                    className="result-badge"
                                    style={{
                                        borderColor: color,
                                        borderWidth: 2,
                                    }}
                                >
                                    {/* Emoji — bouncy spring */}
                                    <motion.span
                                        style={{
                                            fontSize: '3.5rem',
                                            lineHeight: 1,
                                        }}
                                        initial={{ scale: 0, rotate: -20 }}
                                        animate={{ scale: 1, rotate: 0 }}
                                        transition={{
                                            type: 'spring',
                                            stiffness: 380,
                                            damping: 15,
                                            delay: 0.05,
                                        }}
                                    >
                                        {EMOTION_EMOJI[
                                            result.predicted_emotion
                                        ] ?? '🔍'}
                                    </motion.span>

                                    {/* Emotion label */}
                                    <motion.div
                                        className="result-emotion"
                                        style={{ color }}
                                        initial={{ opacity: 0, y: 8 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{
                                            delay: 0.2,
                                            duration: 0.4,
                                            ease: [0.16, 1, 0.3, 1],
                                        }}
                                    >
                                        {result.predicted_emotion}
                                    </motion.div>

                                    {/* Confidence arc */}
                                    <ConfidenceArc
                                        value={result.confidence}
                                        color={color}
                                    />

                                    {/* Confidence label */}
                                    <motion.div
                                        className="result-confidence"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{
                                            delay: 0.38,
                                            duration: 0.4,
                                        }}
                                    >
                                        Confidence score
                                    </motion.div>
                                </div>

                                {result.is_oov && (
                                    <motion.div
                                        className="oov-notice"
                                        initial={{ opacity: 0, y: 6 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{
                                            delay: 0.45,
                                            duration: 0.35,
                                        }}
                                    >
                                        ⚠ Some words weren't in the vocabulary.
                                        The system applied normalisation — check
                                        spelling for best results.
                                    </motion.div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>
            </div>

            {/* ── Probability bars ── */}
            <AnimatePresence>
                {result && (
                    <motion.div
                        className="card"
                        style={{ marginTop: '1.75rem' }}
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 12 }}
                        transition={{
                            opacity: {
                                duration: 0.5,
                                delay: 0.1,
                                ease: 'easeOut',
                            },
                            y: {
                                duration: 0.55,
                                delay: 0.1,
                                ease: [0.16, 1, 0.3, 1],
                            },
                        }}
                    >
                        <p className="section-title">
                            Probability Distribution
                        </p>
                        <ProbabilityBars probabilities={result.probabilities} />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
