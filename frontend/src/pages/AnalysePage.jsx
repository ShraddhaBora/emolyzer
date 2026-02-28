// pages/AnalysePage.jsx
import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiPredict } from '../api'
import ProbabilityBars from '../components/ProbabilityBars'

const EMOTION_EMOJI = {
    Sadness: '😢', Joy: '😄', Love: '❤️', Anger: '😠',
    Fear: '😨', Surprise: '😲', Neutral: '😐',
    'Unknown (Out of Vocabulary)': '🤷',
}

const EXAMPLE_INPUTS = [
    'I can\'t believe how wonderful today has been!',
    'This is absolutely infuriating, I hate it.',
    'I miss my family so much it hurts.',
    'Oh wow, I did not see that coming at all.',
]

export default function AnalysePage({ apiReady }) {
    const [text, setText] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    async function handleSubmit(e) {
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

    function handleExample(ex) {
        setText(ex)
        setResult(null)
        setError(null)
    }

    const color = result?.probabilities?.[result.predicted_emotion]?.color ?? '#6c63ff'

    return (
        <div className="container" style={{ paddingTop: '2rem', paddingBottom: '3rem' }}>

            {/* Hero */}
            <section className="hero" style={{ paddingTop: 0 }}>
                <h1>Understand the <span>emotion</span> behind any text</h1>
                <p>
                    Enter any sentence, message, or phrase. Our multi-model pipeline classifies
                    it into one of seven core emotions with calibrated confidence scores.
                </p>
            </section>

            {/* Example chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', marginBottom: '1.75rem' }}>
                {EXAMPLE_INPUTS.map((ex) => (
                    <button
                        key={ex}
                        onClick={() => handleExample(ex)}
                        style={{
                            background: 'var(--surface)', border: '1px solid var(--border)',
                            borderRadius: 'var(--r-full)', padding: '0.38rem 0.9rem',
                            fontSize: '0.8rem', color: 'var(--text-secondary)',
                            cursor: 'pointer', transition: 'border-color 0.2s',
                        }}
                        onMouseEnter={e => e.target.style.borderColor = 'var(--accent)'}
                        onMouseLeave={e => e.target.style.borderColor = 'var(--border)'}
                    >
                        {ex.length > 38 ? ex.slice(0, 38) + '…' : ex}
                    </button>
                ))}
            </div>

            {/* Main grid */}
            <div className="predictor-wrap">
                {/* Input side */}
                <form onSubmit={handleSubmit} className="card">
                    <p className="section-title">Your Text</p>
                    <textarea
                        id="text-input"
                        className="predictor-textarea"
                        rows={5}
                        placeholder="Type or paste any text here…"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                    {!apiReady && (
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                            ⏳ Model is still training, please wait…
                        </p>
                    )}
                    <button
                        id="analyse-btn"
                        type="submit"
                        className="btn-primary"
                        disabled={!text.trim() || loading || !apiReady}
                    >
                        {loading ? (
                            <><span style={{ width: 16, height: 16, border: '2px solid #fff8', borderTopColor: '#fff', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.7s linear infinite' }} /> Analysing…</>
                        ) : 'Analyse Emotion'}
                    </button>
                    {error && (
                        <div className="oov-notice" style={{ marginTop: '0.75rem', background: '#fff0f3', borderColor: 'var(--danger)', color: 'var(--danger)' }}>
                            ⚠ {error}
                        </div>
                    )}
                </form>

                {/* Result side */}
                <div>
                    <AnimatePresence mode="wait">
                        {!result && !loading && (
                            <motion.div
                                key="placeholder"
                                className="result-badge"
                                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            >
                                <span style={{ fontSize: '2.5rem' }}>✦</span>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                                    Analysis will appear here
                                </p>
                            </motion.div>
                        )}
                        {loading && (
                            <motion.div
                                key="loading"
                                className="result-badge"
                                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            >
                                <div className="spinner-ring" />
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>Running analysis…</p>
                            </motion.div>
                        )}
                        {result && !loading && (
                            <motion.div
                                key="result"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.35 }}
                            >
                                <div className="result-badge" style={{ borderColor: color, borderWidth: 2 }}>
                                    <span style={{ fontSize: '3rem' }}>
                                        {EMOTION_EMOJI[result.predicted_emotion] ?? '✦'}
                                    </span>
                                    <div className="result-emotion" style={{ color }}>
                                        {result.predicted_emotion}
                                    </div>
                                    <div className="result-confidence">
                                        Confidence
                                        <span className="confidence-ring" style={{ color }}>
                                            {' '}{(result.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                                {result.is_oov && (
                                    <div className="oov-notice">
                                        ⚠ Some words were not recognised (out-of-vocabulary). Try correcting any typos for a more accurate prediction.
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* Probability distribution */}
            {result && (
                <motion.div
                    className="card"
                    style={{ marginTop: '1.5rem' }}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.15 }}
                >
                    <p className="section-title">Probability Distribution</p>
                    <ProbabilityBars probabilities={result.probabilities} />
                </motion.div>
            )}
        </div>
    )
}
