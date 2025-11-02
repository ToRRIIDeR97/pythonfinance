import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const apiPrefix = import.meta.env.VITE_API_PREFIX || '/api/v1'

export const http = axios.create({
  baseURL,
  timeout: 15000,
})

export async function listIndices() {
  const { data } = await http.get(`${apiPrefix}/markets/indices`)
  return data
}

export async function listSectorETFs() {
  const { data } = await http.get(`${apiPrefix}/sectors/etfs`)
  return data
}

export async function getIndexHistorical(ticker, { period = '1y', interval = '1d' } = {}) {
  const { data } = await http.get(`${apiPrefix}/markets/indices/${encodeURIComponent(ticker)}/historical`, { params: { period, interval } })
  return data
}

export async function getSectorHistorical(ticker, { period = '1y', interval = '1d' } = {}) {
  const { data } = await http.get(`${apiPrefix}/sectors/etfs/${encodeURIComponent(ticker)}/historical`, { params: { period, interval } })
  return data
}

export async function getSMA(ticker, { period = '1y', interval = '1d', window = 14 } = {}) {
  const { data } = await http.get(`${apiPrefix}/indicators/${encodeURIComponent(ticker)}/sma`, { params: { period, interval, window } })
  return data
}

export async function getSummary(ticker, { period = '1y', interval = '1d' } = {}) {
  const { data } = await http.get(`${apiPrefix}/summary/${encodeURIComponent(ticker)}`, { params: { period, interval } })
  return data
}

export async function getRSI(ticker, { period = '1y', interval = '1d', window = 14 } = {}) {
  const { data } = await http.get(`${apiPrefix}/indicators/${encodeURIComponent(ticker)}/rsi`, { params: { period, interval, window } })
  return data
}

export async function getMACD(ticker, { period = '1y', interval = '1d', fast = 12, slow = 26, signal = 9 } = {}) {
  const { data } = await http.get(`${apiPrefix}/indicators/${encodeURIComponent(ticker)}/macd`, { params: { period, interval, fast, slow, signal } })
  return data
}

export async function getHistoricalData(ticker, category, { period = '30d', interval = '1d' } = {}) {
  let endpoint
  if (category === 'index') {
    endpoint = `${apiPrefix}/markets/indices/${encodeURIComponent(ticker)}/historical`
  } else if (category === 'etf') {
    endpoint = `${apiPrefix}/sectors/etfs/${encodeURIComponent(ticker)}/historical`
  } else {
    // Default to index if category is not specified
    endpoint = `${apiPrefix}/markets/indices/${encodeURIComponent(ticker)}/historical`
  }
  
  const { data } = await http.get(endpoint, { params: { period, interval } })
  return data
}

export async function getSummaryData(ticker, _category) {
  // Backend exposes a unified summary endpoint regardless of category
  const endpoint = `${apiPrefix}/summary/${encodeURIComponent(ticker)}`
  const { data } = await http.get(endpoint)
  return data
}
