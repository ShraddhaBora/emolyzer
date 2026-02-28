/* hooks/useApi.js — Thin API client */
const BASE = '/api'

export async function apiPredict(text) {
    const res = await fetch(`${BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    })
    if (!res.ok) throw new Error((await res.json()).detail ?? 'Prediction failed')
    return res.json()
}

export async function apiHealth() {
    const res = await fetch(`${BASE}/health`)
    if (!res.ok) throw new Error('API unreachable')
    return res.json()
}

export async function apiDistribution() {
    const res = await fetch(`${BASE}/distribution`)
    if (!res.ok) throw new Error('Failed to load distribution')
    return res.json()
}

export async function apiMetrics() {
    const res = await fetch(`${BASE}/metrics`)
    if (!res.ok) throw new Error('Failed to load metrics')
    return res.json()
}

export async function apiClasses() {
    const res = await fetch(`${BASE}/classes`)
    if (!res.ok) throw new Error('Failed to load classes')
    return res.json()
}

export async function apiSamples({ page = 1, pageSize = 20, emotion = 'All', search = '' }) {
    const params = new URLSearchParams({ page, page_size: pageSize })
    if (emotion && emotion !== 'All') params.set('emotion', emotion)
    if (search && search.trim()) params.set('search', search.trim())
    const res = await fetch(`${BASE}/samples?${params}`)
    if (!res.ok) throw new Error('Failed to load samples')
    return res.json()
}

export async function apiAnalyseUpload(file) {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/upload/analyse`, { method: 'POST', body: form })
    if (!res.ok) throw new Error((await res.json()).detail ?? 'Upload analysis failed')
    return res.json()
}

export async function apiRetrainUpload(file) {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/upload/retrain`, { method: 'POST', body: form })
    if (!res.ok) throw new Error((await res.json()).detail ?? 'Retrain failed to start')
    return res.json()
}
