// App.jsx — Root application with tab routing
import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import TrainingBanner from './components/TrainingBanner'
import AnalysePage from './pages/AnalysePage'
import DistributionPage from './pages/DistributionPage'
import MetricsPage from './pages/MetricsPage'
import { apiHealth } from './api'

export default function App() {
    const [activeTab, setActiveTab] = useState('predict')
    const [health, setHealth] = useState(null)

    // Poll the health endpoint until the model is ready
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
            <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

            <main>
                <div className="container" style={{ paddingTop: '1.25rem' }}>
                    <TrainingBanner champion={health?.champion} />
                </div>

                {activeTab === 'predict' && <AnalysePage apiReady={apiReady} />}
                {activeTab === 'distribution' && <DistributionPage />}
                {activeTab === 'metrics' && <MetricsPage />}
            </main>

            <footer className="footer">
                <p>
                    Emolyzer — Emotion Classification Research System &nbsp;·&nbsp;
                    7-class Ekman model · ~470K training samples · Logistic Regression + Naïve Bayes + Linear SVM
                </p>
            </footer>
        </>
    )
}
