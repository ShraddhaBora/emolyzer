// pages/DistributionPage.jsx
import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { apiDistribution } from '../api'

import DatasetViewer from '../components/DatasetViewer'
import DatasetUploader from '../components/DatasetUploader'

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

    if (loading) return (
        <div className="spinner">
            <div className="spinner-ring" />
            <p>Loading dataset…</p>
        </div>
    )
    if (error) return <div className="container" style={{ padding: '2rem' }}><div className="oov-notice">{error}</div></div>

    const dist = data.distribution ?? []

    return (
        <div className="container" style={{ paddingTop: '2rem', paddingBottom: '3rem' }}>
            <section className="hero" style={{ paddingTop: 0 }}>
                <h1>Dataset <span>Distribution</span></h1>
                <p>
                    The training corpus spans {data.total?.toLocaleString()} labelled examples across {data.num_classes} emotion classes,
                    sourced from Twitter datasets, GoEmotions (Reddit), and the DAIR-AI emotion corpus.
                </p>
            </section>

            {/* Summary tiles */}
            <div className="stat-grid" style={{ marginBottom: '1.75rem' }}>
                <div className="stat-tile">
                    <div className="value">{(data.total / 1000).toFixed(0)}K</div>
                    <div className="label">Total Samples</div>
                </div>
                <div className="stat-tile">
                    <div className="value">{data.num_classes}</div>
                    <div className="label">Emotion Classes</div>
                </div>
                <div className="stat-tile">
                    <div className="value">3</div>
                    <div className="label">Source Datasets</div>
                </div>
            </div>

            {/* Bar chart */}
            <motion.div
                className="card"
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <p className="section-title">Sample Count per Emotion</p>
                <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                        <BarChart data={dist} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                            <XAxis
                                dataKey="Emotion"
                                tick={{ fontSize: 12, fill: 'var(--text-secondary)', fontFamily: 'Inter' }}
                                axisLine={false} tickLine={false}
                            />
                            <YAxis
                                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
                                tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'Inter' }}
                                axisLine={false} tickLine={false} width={44}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: 'var(--surface)', border: '1px solid var(--border)',
                                    borderRadius: 10, fontSize: 13, fontFamily: 'Inter',
                                }}
                                formatter={(v) => [v.toLocaleString(), 'Samples']}
                            />
                            <Bar dataKey="Count" radius={[6, 6, 0, 0]}>
                                {dist.map((entry) => (
                                    <Cell key={entry.Emotion} fill={entry.Color ?? '#6c63ff'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* Class breakdown table */}
            <motion.div
                className="card"
                style={{ marginTop: '1.5rem' }}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1 }}
            >
                <p className="section-title">Breakdown</p>
                <table className="metrics-table">
                    <thead>
                        <tr>
                            <th>Emotion</th>
                            <th>Count</th>
                            <th>Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dist.map((row) => (
                            <tr key={row.Emotion}>
                                <td>
                                    <span style={{
                                        display: 'inline-block', width: 10, height: 10,
                                        borderRadius: '50%', background: row.Color ?? '#ccc',
                                        marginRight: '0.5rem', verticalAlign: 'middle',
                                    }} />
                                    {row.Emotion}
                                </td>
                                <td>{row.Count?.toLocaleString()}</td>
                                <td>{row.Percentage?.toFixed(1)}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </motion.div>

            <DatasetViewer />
            <DatasetUploader />
        </div>
    )
}
