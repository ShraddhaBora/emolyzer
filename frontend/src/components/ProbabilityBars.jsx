// components/ProbabilityBars.jsx
import React from 'react'
import { motion } from 'framer-motion'

export default function ProbabilityBars({ probabilities }) {
    if (!probabilities) return null

    const entries = Object.entries(probabilities)
        .map(([emotion, data]) => ({ emotion, ...data }))
        .sort((a, b) => b.probability - a.probability)

    return (
        <div className="prob-list">
            {entries.map(({ emotion, probability, color }, i) => (
                <motion.div
                    key={emotion}
                    className="prob-row"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05, duration: 0.3 }}
                >
                    <span className="prob-label">{emotion}</span>
                    <div className="prob-bar-track">
                        <motion.div
                            className="prob-bar-fill"
                            style={{ background: color }}
                            initial={{ width: 0 }}
                            animate={{ width: `${(probability * 100).toFixed(1)}%` }}
                            transition={{ duration: 0.7, delay: i * 0.04, ease: 'easeOut' }}
                        />
                    </div>
                    <span className="prob-pct">{(probability * 100).toFixed(1)}%</span>
                </motion.div>
            ))}
        </div>
    )
}
