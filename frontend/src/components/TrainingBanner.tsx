// components/TrainingBanner.tsx
import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface TrainingBannerProps {
    champion: string | null | undefined
}

export default function TrainingBanner({ champion }: TrainingBannerProps) {
    const ready = champion && champion !== 'not trained'

    return (
        <AnimatePresence mode="wait">
            <motion.div
                key={ready ? 'ready' : 'loading'}
                className={`status-banner ${ready ? 'ready' : 'loading'}`}
                initial={{ opacity: 0, y: -6, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.98 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
            >
                <span className={`status-dot ${ready ? 'ready' : 'pulse'}`} />
                {ready ? (
                    <>
                        Selected Model:&nbsp;<strong>{champion}</strong>
                        <span
                            style={{
                                marginLeft: 'auto',
                                fontSize: '0.75rem',
                                opacity: 0.7,
                                fontWeight: 400,
                            }}
                        >
                            5-Fold CV Champion
                        </span>
                    </>
                ) : (
                    <>
                        <span>Training pipeline</span>
                        <span
                            style={{
                                display: 'flex',
                                gap: 4,
                                alignItems: 'center',
                            }}
                        >
                            <span className="spinner-dots">
                                <span />
                                <span />
                                <span />
                            </span>
                        </span>
                    </>
                )}
            </motion.div>
        </AnimatePresence>
    )
}
