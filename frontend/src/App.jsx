// App.jsx — Root application with tab routing
import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import TrainingBanner from './components/TrainingBanner'
import AnalysePage from './pages/AnalysePage'
import DistributionPage from './pages/DistributionPage'
import MetricsPage from './pages/MetricsPage'
import { apiHealth } from './api'

// Floating Action Button — scroll to top
function FAB() {
    const [show, setShow] = useState(false)
    useEffect(() => {
        const fn = () => setShow(window.scrollY > 300)
        window.addEventListener('scroll', fn, { passive: true })
        return () => window.removeEventListener('scroll', fn)
    }, [])
    return (
        <AnimatePresence>
            {show && (
                <motion.button
                    className="fab"
                    title="Back to top"
                    onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                    initial={{ opacity: 0, scale: 0.5, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.5, y: 20 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 22 }}
                    whileHover={{ scale: 1.12 }}
                    whileTap={{ scale: 0.92 }}
                >
                    ↑
                </motion.button>
            )}
        </AnimatePresence>
    )
}

const pageVariants = {
    initial: {
        opacity: 0,
    },
    animate: {
        opacity: 1,
        transition: {
            duration: 0.3,
            ease: 'easeOut',
        },
    },
    exit: {
        opacity: 0,
        transition: {
            duration: 0.18,
            ease: 'easeIn',
        },
    },
}

export default function App() {
    const [activeTab, setActiveTab] = useState('predict')
    const [health, setHealth] = useState(null)

    function handleTabChange(tab) {
        setActiveTab(tab)
    }

    useEffect(() => {
        let timer
        async function check() {
            try {
                const h = await apiHealth()
                setHealth(h)
                if (h.champion === 'not trained') {
                    timer = setTimeout(check, 4000)
                }
            } catch {
                timer = setTimeout(check, 5000)
            }
        }
        check()
        return () => clearTimeout(timer)
    }, [])

    const apiReady = health && health.champion !== 'not trained'

    return (
        <>
            <Navbar activeTab={activeTab} onTabChange={handleTabChange} />

            <main style={{ overflow: 'hidden' }}>
                <div className="container" style={{ paddingTop: '1.25rem' }}>
                    <TrainingBanner champion={health?.champion} />
                </div>

                <AnimatePresence mode="wait">
                    {activeTab === 'predict' && (
                        <motion.div key="predict"
                            variants={pageVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                        >
                            <AnalysePage apiReady={apiReady} />
                        </motion.div>
                    )}
                    {activeTab === 'distribution' && (
                        <motion.div key="distribution"
                            variants={pageVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                        >
                            <DistributionPage />
                        </motion.div>
                    )}
                    {activeTab === 'metrics' && (
                        <motion.div key="metrics"
                            variants={pageVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                        >
                            <MetricsPage />
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>

            <footer className="footer">
                <p>
                    Emolyzer - Emotion Classification Research&nbsp;·&nbsp;
                    7-class Ekman model&nbsp;·&nbsp;~470K training samples
                </p>
            </footer>

            <FAB />
        </>
    )
}
