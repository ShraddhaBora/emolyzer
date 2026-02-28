/* components/DatasetViewer.jsx */
import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { apiClasses, apiSamples } from '../api'

export default function DatasetViewer() {
    const [samples, setSamples] = useState([])
    const [total, setTotal] = useState(0)
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)

    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    const [emotions, setEmotions] = useState([])
    const [filterEmotion, setFilterEmotion] = useState('All')
    const [search, setSearch] = useState('')

    useEffect(() => {
        apiClasses().then(res => setEmotions(res.emotions)).catch(() => { })
    }, [])

    useEffect(() => {
        const fetchSamples = async () => {
            setLoading(true)
            try {
                const res = await apiSamples({ page, emotion: filterEmotion, search })
                setSamples(res.samples)
                setTotal(res.total)
                setTotalPages(res.total_pages)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        // Debounce search slightly
        const timer = setTimeout(fetchSamples, 300)
        return () => clearTimeout(timer)
    }, [page, filterEmotion, search])

    const handleFilterChange = (e) => {
        setFilterEmotion(e.target.value)
        setPage(1)
    }

    const handleSearchChange = (e) => {
        setSearch(e.target.value)
        setPage(1)
    }

    return (
        <motion.div
            className="card"
            style={{ marginTop: '2rem' }}
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}
        >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                    <p className="section-title" style={{ margin: 0 }}>Browse Training Data</p>
                    <p style={{ margin: 0, fontSize: '0.85rem', color: '#64748b' }}>{total.toLocaleString()} total labeled samples</p>
                </div>

                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <input
                        type="text"
                        placeholder="Search dataset..."
                        value={search}
                        onChange={handleSearchChange}
                        className="input-field"
                        style={{ padding: '0.5rem 0.75rem', width: 220, margin: 0 }}
                    />
                    <select
                        value={filterEmotion}
                        onChange={handleFilterChange}
                        style={{ padding: '0.5rem 0.75rem', borderRadius: 12, border: '1px solid #e2e8f0', background: '#f8fafc', color: '#475569', fontFamily: 'Inter', outline: 'none' }}
                    >
                        <option value="All">All Emotions</option>
                        {emotions.map(em => (
                            <option key={em.id} value={em.name}>{em.name}</option>
                        ))}
                    </select>
                </div>
            </div>

            {error ? (
                <div className="oov-notice">{error}</div>
            ) : (
                <>
                    <table className="metrics-table" style={{ opacity: loading ? 0.5 : 1, transition: 'opacity 0.2s' }}>
                        <thead>
                            <tr>
                                <th style={{ width: '80%' }}>Text Sample</th>
                                <th style={{ width: '20%' }}>Label</th>
                            </tr>
                        </thead>
                        <tbody>
                            {samples.length === 0 && !loading && (
                                <tr>
                                    <td colSpan="2" style={{ textAlign: 'center', color: '#94a3b8', padding: '2rem' }}>No samples matched your search.</td>
                                </tr>
                            )}
                            {samples.map((row, idx) => {
                                const color = emotions.find(e => e.name === row.emotion)?.color ?? '#ccc'
                                return (
                                    <tr key={idx}>
                                        <td style={{ color: '#334155' }}>"{row.text}"</td>
                                        <td>
                                            <span style={{
                                                display: 'inline-block', padding: '0.2rem 0.6rem',
                                                borderRadius: 12, fontSize: '0.75rem', fontWeight: 600,
                                                background: `${color}20`, color: color
                                            }}>
                                                {row.emotion}
                                            </span>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem', borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
                        <button
                            className="btn btn-secondary"
                            style={{ padding: '0.4rem 1rem' }}
                            disabled={page === 1}
                            onClick={() => setPage(p => p - 1)}
                        >
                            Previous
                        </button>
                        <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
                            Page <strong>{page}</strong> of <strong>{totalPages.toLocaleString()}</strong>
                        </span>
                        <button
                            className="btn btn-secondary"
                            style={{ padding: '0.4rem 1rem' }}
                            disabled={page >= totalPages}
                            onClick={() => setPage(p => p + 1)}
                        >
                            Next
                        </button>
                    </div>
                </>
            )}
        </motion.div>
    )
}
