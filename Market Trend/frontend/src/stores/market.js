import { defineStore } from 'pinia'

export const useMarketStore = defineStore('market', {
  state: () => ({
    selectedCategory: 'index',
    selectedTicker: '^GSPC',
    timeframe: { period: '1y', interval: '1d' },
    indicators: [
      { type: 'sma', window: 14, color: '#ff7f0e' }
    ],
  }),
  actions: {
    setSelection({ category, ticker }) {
      if (category) this.selectedCategory = category
      if (ticker) this.selectedTicker = ticker
    },
    setTimeframe(tf) {
      this.timeframe = tf
    },
    addIndicator(indicator) {
      const exists = this.indicators.some(
        (ind) => ind.type === indicator.type && ind.window === indicator.window
      )
      if (!exists) {
        this.indicators.push(indicator)
      }
    },
    removeIndicator(indicator) {
      this.indicators = this.indicators.filter(
        (ind) => !(ind.type === indicator.type && ind.window === indicator.window)
      )
    },
    setIndicators(list) {
      this.indicators = Array.isArray(list) ? list : []
    },
  }
})
