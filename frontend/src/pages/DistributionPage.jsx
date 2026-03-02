// pages/DistributionPage.jsx
import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { apiDistribution } from '../api'
import DatasetViewer from '../components/DatasetViewer'
import DatasetUploader from '../components/DatasetUploader'

const f = (delay) => ({
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: 0.6, delay, ease: 'easeOut' },
})

// Spinner shown inside AnimatePresence so it cross-fades with content
function LoadingState({ label }) {
    return (
        <motion.div
            key="loading"
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '1rem' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
        >
            <div className="spinner-ring" />
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>{label}</p>
        </motion.div>
    )
}

export default function DistributionPage() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        apiDistribution()
            .then(setData)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false))
    }, [])

    const dist = data?.distribution ?? []

    return (
        <div className="container" style={{ paddingTop: '2rem', paddingBottom: '3rem' }}>
            <AnimatePresence mode="wait">

                {/* Loading — fades out before content fades in */}
                {loading && <LoadingState key="loading" label="Loading dataset…" />}

                {/* Error */}
                {!loading && error && (
                    <motion.div key="error" {...f(0.05)}>
                        <div className="oov-notice">{error}</div>
                    </motion.div>
                )}

                {/* Content — fades in after spinner exits */}
                {!loading && !error && data && (
                    <motion.div
                        key="content"
                        initial={false}
                        animate={{}}
                    >
                        {/* Hero */}
                        <section className="hero" style={{ paddingTop: 0 }}>
                            <motion.h1 {...f(0.05)}>
                                Dataset <span>Distribution</span>
                            </motion.h1>
                            <motion.p {...f(0.18)}>
                                The training corpus spans {data.total?.toLocaleString()} labelled examples
                                across {data.num_classes} emotion classes, sourced from Twitter datasets,
                                GoEmotions (Reddit), and the DAIR-AI emotion corpus.
                            </motion.p>
                        </section>

                        {/* Stat tiles — staggered individually */}
                        <div className="stat-grid" style={{ marginBottom: '1.75rem' }}>
                            {[
                                { value: `${(data.total / 1000).toFixed(0)}K`, label: 'Total Samples', d: 0.30 },
                                { value: data.num_classes, label: 'Emotion Classes', d: 0.35 },
                                { value: 3, label: 'Source Datasets', d: 0.40 },
                            ].map(({ value, label, d }) => (
                                <motion.div key={label} {...f(d)} className="stat-tile">
                                    <div className="value">{value}</div>
                                    <div className="label">{label}</div>
                                </motion.div>
                            ))}
                        </div>

                        {/* Bar chart */}
                        <motion.div {...f(0.50)} className="card">
                            <p className="section-title">Sample Count per Emotion</p>
                            <div style={{ width: '100%', height: 300 }}>
                                <ResponsiveContainer>
                                    <BarChart data={dist} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                                        <XAxis dataKey="Emotion" tick={{ fontSize: 12, fill: 'var(--text-secondary)', fontFamily: 'Inter' }} axisLine={false} tickLine={false} />
                                        <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'Inter' }} axisLine={false} tickLine={false} width={44} />
                                        <Tooltip
                                            cursor={{ fill: '#fcfaf2', opacity: 0.7 }}
                                            contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 13, fontFamily: 'Inter' }}
                                            formatter={(v) => [v.toLocaleString(), 'Samples']}
                                        />
                                        <Bar dataKey="Count" radius={[6, 6, 0, 0]} background={{ fill: '#f8f7f2', radius: [6, 6, 0, 0] }}>
                                            {dist.map((entry) => <Cell key={entry.Emotion} fill={entry.Color ?? '#6c63ff'} />)}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </motion.div>

                        {/* Breakdown table */}
                        <motion.div {...f(0.62)} className="card" style={{ marginTop: '1.5rem' }}>
                            <p className="section-title">Breakdown</p>
                            <table className="metrics-table">
                                <thead>
                                    <tr><th>Emotion</th><th>Count</th><th>Share</th></tr>
                                </thead>
                                <tbody>
                                    {dist.map((row, i) => (
                                        <motion.tr
                                            key={row.Emotion}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: 0.70 + i * 0.05, duration: 0.4, ease: 'easeOut' }}
                                        >
                                            <td>
                                                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: row.Color ?? '#ccc', marginRight: '0.5rem', verticalAlign: 'middle' }} />
                                                {row.Emotion}
                                            </td>
                                            <td>{row.Count?.toLocaleString()}</td>
                                            <td>{row.Percentage?.toFixed(1)}%</td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </motion.div>

                        <DatasetViewer />
                        <DatasetUploader />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
