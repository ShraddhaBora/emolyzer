/* components/DatasetUploader.jsx */
import React, { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { apiAnalyseUpload, apiRetrainUpload } from '../api'

export default function DatasetUploader() {
    const [file, setFile] = useState(null)
    const [analysis, setAnalysis] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [retrainStarted, setRetrainStarted] = useState(false)
    const fileInputRef = useRef(null)

    const handleFileChange = async (e) => {
        const selected = e.target.files[0]
        if (!selected) return
        setFile(selected)
        setLoading(true)
        setError(null)
        setAnalysis(null)
        try {
            const result = await apiAnalyseUpload(selected)
            setAnalysis(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
            if (fileInputRef.current) fileInputRef.current.value = ''
        }
    }

    const handleRetrain = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        try {
            await apiRetrainUpload(file)
            setRetrainStarted(true)
        } catch (err) {
            setError(err.message)
            setLoading(false)
        }
    }

    return (
        <motion.div
            className="card upload-section"
            style={{ marginTop: '2rem', background: '#f8fafc', border: '2px dashed #e2e8f0' }}
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
        >
            <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
                <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e293b' }}>Upload Custom Dataset</h3>
                <p style={{ color: '#64748b', fontSize: '0.95rem', margin: '0 0 1.5rem 0' }}>
                    Upload a CSV with <code>text</code> and <code>label</code> (0–6) columns.
                    The model will train <strong>exclusively</strong> on this file — completely independent of the default dataset.
                </p>

                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                    ref={fileInputRef}
                />

                {!loading && !analysis && !retrainStarted && (
                    <button
                        className="btn btn-secondary"
                        onClick={() => fileInputRef.current?.click()}
                    >
                        Select CSV File
                    </button>
                )}

                {loading && <div className="spinner-ring" style={{ margin: '0 auto' }} />}
                {error && <div className="oov-notice" style={{ marginTop: '1rem', display: 'inline-block' }}>{error}</div>}

                <AnimatePresence>
                    {analysis && !retrainStarted && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                            style={{ textAlign: 'left', marginTop: '1.5rem', background: 'white', padding: '1.5rem', borderRadius: 12, border: '1px solid #e2e8f0' }}
                        >
                            <h4 style={{ margin: '0 0 1rem 0' }}>File: <span style={{ color: '#6c63ff' }}>{analysis.filename}</span></h4>
                            <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
                                <div style={{ background: '#f1f5f9', padding: '0.75rem', borderRadius: 8 }}>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{analysis.total_rows.toLocaleString()}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Valid Rows</div>
                                </div>
                                <div style={{ background: '#f1f5f9', padding: '0.75rem', borderRadius: 8 }}>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{analysis.num_classes}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Classes Found</div>
                                </div>
                                <div style={{ background: analysis.unmapped_labels > 0 ? '#fee2e2' : '#f1f5f9', padding: '0.75rem', borderRadius: 8 }}>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600, color: analysis.unmapped_labels > 0 ? '#ef4444' : 'inherit' }}>{analysis.unmapped_labels}</div>
                                    <div style={{ fontSize: '0.8rem', color: analysis.unmapped_labels > 0 ? '#ef4444' : '#64748b' }}>Unmapped Labels</div>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem', justifyContent: 'flex-end' }}>
                                <button className="btn btn-secondary" onClick={() => { setAnalysis(null); setFile(null) }}>Cancel</button>
                                <button className="btn" onClick={handleRetrain} disabled={loading || analysis.unmapped_labels > 0}>
                                    Replace Dataset & Retrain
                                </button>
                            </div>
                        </motion.div>
                    )}

                    {retrainStarted && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '1rem', color: '#0d6e52', fontWeight: 500 }}>
                            ✅ Retraining initiated in backend! The "Selected Model" banner will update once completed.
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.div>
    )
}
