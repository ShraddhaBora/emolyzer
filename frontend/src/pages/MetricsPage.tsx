// @ts-nocheck
// pages/MetricsPage.jsx
import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    Radar,
    ResponsiveContainer,
    Tooltip,
} from 'recharts'
import { apiMetrics } from '../api'

const f = (delay: number) => ({
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: 0.6, delay, ease: 'easeOut' },
})

function LoadingState({ label }: { label: string }) {
    return (
        <motion.div
            key="loading"
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '60vh',
                gap: '1rem',
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
        >
            <div className="spinner-ring" />
            <p
                style={{
                    color: 'var(--text-muted)',
                    fontSize: '0.9rem',
                    fontFamily: 'var(--font-mono)',
                }}
            >
                {label}
            </p>
        </motion.div>
    )
}

export default function MetricsPage() {
    const [data, setData] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        apiMetrics()
            .then(setData)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false))
    }, [])

    const cv = data?.cross_validation ?? []
    const radarData = cv.map((m: any) => ({
        model: m.model,
        'Mean F1': parseFloat((m.mean_f1 * 100).toFixed(2)),
    }))

    const methodItems = [
        {
            title: 'Preprocessing',
            body: 'Conjunction-aware negation marking, synonym injection for high-intensity emotion words, and bigram TF-IDF vectorisation with sublinear scaling.',
        },
        {
            title: 'Model Candidates',
            body: 'Logistic Regression (L2), Multinomial Naïve Bayes, and Linear SVM with Platt scaling — all evaluated on equal footing.',
        },
        {
            title: 'Evaluation',
            body: '5-Fold Stratified Cross-Validation on the training set ensures stable, non-optimistic performance estimates before holdout evaluation.',
        },
        {
            title: 'Data Sources',
            body: '~470K samples: Twitter emotion dataset, GoEmotions (Reddit), and DAIR-AI emotion — mapped to 7 Ekman-style classes.',
        },
    ]

    return (
        <div
            className="container"
            style={{ paddingTop: '2rem', paddingBottom: '3rem' }}
        >
            <AnimatePresence mode="wait">
                {loading && (
                    <LoadingState
                        key="loading"
                        label="Loading model metrics…"
                    />
                )}

                {!loading && error && (
                    <motion.div key="error" {...f(0.05)}>
                        <div className="oov-notice">{error}</div>
                    </motion.div>
                )}

                {!loading && !error && data && (
                    <motion.div key="content" initial={false} animate={{}}>
                        {/* Hero */}
                        <section className="hero" style={{ paddingTop: 0 }}>
                            <motion.h1 {...f(0.05)}>
                                Model <span>Performance</span>
                            </motion.h1>
                            <motion.p {...f(0.18)}>
                                Three algorithms are evaluated under identical
                                5-fold stratified cross-validation. The best
                                performer is selected as the champion model for
                                live inference.
                            </motion.p>
                        </section>

                        {/* Stat tiles — staggered individually */}
                        <div
                            className="stat-grid"
                            style={{ marginBottom: '1.75rem' }}
                        >
                            {[
                                {
                                    value: data.champion_model,
                                    label: 'Selected Model',
                                    d: 0.3,
                                },
                                {
                                    value: `${(data.holdout_accuracy * 100).toFixed(1)}%`,
                                    label: 'Holdout Accuracy',
                                    d: 0.35,
                                },
                                {
                                    value: data.macro_f1?.toFixed(3),
                                    label: 'Macro F1 Score',
                                    d: 0.4,
                                },
                                {
                                    value: data.weighted_f1?.toFixed(3),
                                    label: 'Weighted F1 Score',
                                    d: 0.45,
                                },
                                {
                                    value:
                                        data.test_samples?.toLocaleString?.() ??
                                        data.test_samples,
                                    label: 'Test Samples',
                                    d: 0.5,
                                },
                            ].map(({ value, label, d }) => (
                                <motion.div
                                    key={label}
                                    {...f(d)}
                                    className="stat-tile"
                                >
                                    <div
                                        className="value"
                                        style={{ fontSize: '1.75rem' }}
                                    >
                                        {value}
                                    </div>
                                    <div className="label">{label}</div>
                                </motion.div>
                            ))}
                        </div>

                        {/* CV table + Radar */}
                        <div
                            style={{
                                display: 'grid',
                                gridTemplateColumns: '1fr 1fr',
                                gap: '1.5rem',
                            }}
                        >
                            <motion.div {...f(0.6)} className="card">
                                <p className="section-title">
                                    Cross-Validation Results
                                </p>
                                <table className="metrics-table">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Mean F1</th>
                                            <th>Std Dev</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {cv.map((m, i) => (
                                            <motion.tr
                                                key={m.model}
                                                className={
                                                    m.model ===
                                                    data.champion_model
                                                        ? 'champion-row'
                                                        : ''
                                                }
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                                transition={{
                                                    delay: 0.7 + i * 0.09,
                                                    duration: 0.4,
                                                    ease: 'easeOut',
                                                }}
                                            >
                                                <td>
                                                    {m.model}
                                                    {m.model ===
                                                        data.champion_model && (
                                                        <span className="badge-champion">
                                                            ★ Selected
                                                        </span>
                                                    )}
                                                </td>
                                                <td>
                                                    {(m.mean_f1 * 100).toFixed(
                                                        2
                                                    )}
                                                    %
                                                </td>
                                                <td>
                                                    ±
                                                    {(m.std_f1 * 100).toFixed(
                                                        2
                                                    )}
                                                    %
                                                </td>
                                            </motion.tr>
                                        ))}
                                    </tbody>
                                </table>
                            </motion.div>

                            <motion.div {...f(0.68)} className="card">
                                <p className="section-title">
                                    F1 Comparison (Radar)
                                </p>
                                <div style={{ width: '100%', height: 240 }}>
                                    <ResponsiveContainer>
                                        <RadarChart data={radarData}>
                                            <PolarGrid stroke="var(--border)" />
                                            <PolarAngleAxis
                                                dataKey="model"
                                                tick={{
                                                    fontSize: 11,
                                                    fill: 'var(--text-secondary)',
                                                    fontFamily: 'Inter',
                                                }}
                                            />
                                            <Radar
                                                dataKey="Mean F1"
                                                stroke="#6c63ff"
                                                fill="#6c63ff"
                                                fillOpacity={0.15}
                                                dot={{ r: 4, fill: '#6c63ff' }}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background:
                                                        'var(--surface)',
                                                    border: '1px solid var(--border)',
                                                    borderRadius: 10,
                                                    fontFamily: 'Inter',
                                                    fontSize: 12,
                                                }}
                                                formatter={(v) => [
                                                    `${v.toFixed(2)}%`,
                                                    'Mean F1',
                                                ]}
                                            />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </motion.div>
                        </div>

                        {/* Methodology */}
                        <motion.div
                            {...f(0.58)}
                            className="card"
                            style={{ marginTop: '1.5rem' }}
                        >
                            <p className="section-title">
                                Research Methodology
                            </p>
                            <div
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns:
                                        'repeat(auto-fit, minmax(220px, 1fr))',
                                    gap: '1rem',
                                    marginTop: '0.5rem',
                                }}
                            >
                                {methodItems.map(({ title, body }, i) => (
                                    <motion.div
                                        key={title}
                                        className="card-sm"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{
                                            delay: 0.65 + i * 0.08,
                                            duration: 0.45,
                                            ease: 'easeOut',
                                        }}
                                    >
                                        <p
                                            style={{
                                                fontWeight: 600,
                                                fontSize: '0.9rem',
                                                marginBottom: '0.4rem',
                                                color: 'var(--text-primary)',
                                            }}
                                        >
                                            {title}
                                        </p>
                                        <p
                                            style={{
                                                fontSize: '0.83rem',
                                                color: 'var(--text-secondary)',
                                                lineHeight: 1.65,
                                            }}
                                        >
                                            {body}
                                        </p>
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
