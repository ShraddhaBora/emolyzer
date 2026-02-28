// pages/MetricsPage.jsx
import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts'
import { apiMetrics } from '../api'

export default function MetricsPage() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        apiMetrics()
            .then(setData)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false))
    }, [])

    if (loading) return (
        <div className="spinner">
            <div className="spinner-ring" />
            <p>Loading model metrics…</p>
        </div>
    )
    if (error) return <div className="container" style={{ padding: '2rem' }}><div className="oov-notice">{error}</div></div>

    const cv = data.cross_validation ?? []

    // Radar data: one entry per model's mean F1
    const radarData = cv.map((m) => ({
        model: m.model,
        'Mean F1': parseFloat((m.mean_f1 * 100).toFixed(2)),
    }))

    return (
        <div className="container" style={{ paddingTop: '2rem', paddingBottom: '3rem' }}>
            <section className="hero" style={{ paddingTop: 0 }}>
                <h1>Model <span>Performance</span></h1>
                <p>
                    Three algorithms are evaluated under identical 5-fold stratified cross-validation.
                    The best performer is selected as the champion model for live inference.
                </p>
            </section>

            {/* Holdout summary tiles */}
            <div className="stat-grid" style={{ marginBottom: '1.75rem' }}>
                {[
                    { value: data.champion_model, label: 'Selected Model' },
                    { value: `${(data.holdout_accuracy * 100).toFixed(1)}%`, label: 'Holdout Accuracy' },
                    { value: data.macro_f1?.toFixed(3), label: 'Macro F1 Score' },
                    { value: data.weighted_f1?.toFixed(3), label: 'Weighted F1 Score' },
                    { value: data.test_samples?.toLocaleString?.() ?? data.test_samples, label: 'Test Samples' },
                ].map(({ value, label }) => (
                    <div key={label} className="stat-tile">
                        <div className="value" style={{ fontSize: '1.75rem' }}>{value}</div>
                        <div className="label">{label}</div>
                    </div>
                ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                {/* CV results table */}
                <motion.div
                    className="card"
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                >
                    <p className="section-title">Cross-Validation Results</p>
                    <table className="metrics-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Mean F1</th>
                                <th>Std Dev</th>
                            </tr>
                        </thead>
                        <tbody>
                            {cv.map((m) => (
                                <tr key={m.model} className={m.model === data.champion_model ? 'champion-row' : ''}>
                                    <td>
                                        {m.model}
                                        {m.model === data.champion_model && (
                                            <span className="badge-champion">★ Selected</span>
                                        )}
                                    </td>
                                    <td>{(m.mean_f1 * 100).toFixed(2)}%</td>
                                    <td>±{(m.std_f1 * 100).toFixed(2)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </motion.div>

                {/* Radar chart */}
                <motion.div
                    className="card"
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.1 }}
                >
                    <p className="section-title">F1 Comparison (Radar)</p>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="var(--border)" />
                                <PolarAngleAxis dataKey="model" tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'Inter' }} />
                                <Radar dataKey="Mean F1" stroke="#6c63ff" fill="#6c63ff" fillOpacity={0.15} dot={{ r: 4, fill: '#6c63ff' }} />
                                <Tooltip
                                    contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, fontFamily: 'Inter', fontSize: 12 }}
                                    formatter={(v) => [`${v.toFixed(2)}%`, 'Mean F1']}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            </div>

            {/* Methodology */}
            <motion.div
                className="card"
                style={{ marginTop: '1.5rem' }}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
            >
                <p className="section-title">Research Methodology</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
                    {[
                        { title: 'Preprocessing', body: 'Conjunction-aware negation marking (NOT_ prefix), unicode normalisation, and bigram TF-IDF vectorisation with sublinear scaling.' },
                        { title: 'Model Candidates', body: 'Logistic Regression (L2), Multinomial Naïve Bayes, and Linear SVM with Platt scaling — all evaluated on equal footing.' },
                        { title: 'Evaluation', body: '5-Fold Stratified Cross-Validation on the training set ensures stable, non-optimistic performance estimates before holdout evaluation.' },
                        { title: 'Data Sources', body: '~470K samples from diverse corpora: Twitter emotion dataset, GoEmotions (Reddit), and DAIR-AI emotion, mapped to 7 Ekman-style classes.' },
                    ].map(({ title, body }) => (
                        <div key={title} className="card-sm">
                            <p style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.4rem', color: 'var(--text-primary)' }}>{title}</p>
                            <p style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>{body}</p>
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    )
}
