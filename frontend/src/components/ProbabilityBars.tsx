// @ts-nocheck
// components/ProbabilityBars.tsx
import React from 'react'
import { motion } from 'framer-motion'

const containerVariants = {
    hidden: {},
    visible: {
        transition: { staggerChildren: 0.07, delayChildren: 0.05 },
    },
}

const rowVariants = {
    hidden: { opacity: 0, x: -18, filter: 'blur(4px)' },
    visible: {
        opacity: 1,
        x: 0,
        filter: 'blur(0px)',
        transition: { type: 'spring', stiffness: 280, damping: 22 },
    },
}

interface ProbabilityBarsProps {
    probabilities:
        | Record<string, { probability: number; color: string }>
        | null
        | undefined
}

export default function ProbabilityBars({
    probabilities,
}: ProbabilityBarsProps) {
    if (!probabilities) return null

    const entries = Object.entries(probabilities)
        .map(([emotion, data]) => ({ emotion, ...data }))
        .sort((a, b) => b.probability - a.probability)

    return (
        <motion.div
            className="prob-list"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            {entries.map(({ emotion, probability, color }, i) => (
                <motion.div
                    key={emotion}
                    className="prob-row"
                    variants={rowVariants}
                >
                    <span className="prob-label">{emotion}</span>
                    <div className="prob-bar-track">
                        <motion.div
                            className="prob-bar-fill"
                            style={{ background: color }}
                            initial={{ width: 0, opacity: 0.6 }}
                            animate={{
                                width: `${(probability * 100).toFixed(1)}%`,
                                opacity: 1,
                            }}
                            transition={{
                                width: {
                                    duration: 0.75,
                                    delay: 0.1 + i * 0.06,
                                    ease: [0.16, 1, 0.3, 1],
                                },
                                opacity: {
                                    duration: 0.3,
                                    delay: 0.1 + i * 0.06,
                                },
                            }}
                        />
                    </div>
                    <motion.span
                        className="prob-pct"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 + i * 0.06 }}
                    >
                        {(probability * 100).toFixed(1)}%
                    </motion.span>
                </motion.div>
            ))}
        </motion.div>
    )
}
