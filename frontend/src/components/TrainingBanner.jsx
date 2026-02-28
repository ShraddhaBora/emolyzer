// components/TrainingBanner.jsx
import React from 'react'

export default function TrainingBanner({ champion }) {
    if (champion && champion !== 'not trained') {
        return (
            <div className="training-banner" style={{ background: '#edfaf5', borderColor: '#22c997', color: '#0d6e52' }}>
                <span className="training-dot" style={{ background: '#22c997' }} />
                Selected Model: <strong>{champion}</strong>
            </div>
        )
    }
    return (
        <div className="training-banner">
            <span className="training-dot" />
            Model is training. This may take a minute on first load...
        </div>
    )
}
