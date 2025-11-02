<template>
  <div class="market-comparison">
    <div class="comparison-header">
      <h3 class="comparison-title">Market Comparison</h3>
      <p class="comparison-subtitle">Compare multiple tickers side by side</p>
    </div>

    <!-- Ticker Selection -->
    <div class="ticker-selection">
      <div class="selected-tickers">
        <div 
          v-for="(ticker, index) in selectedTickers" 
          :key="ticker.symbol"
          class="ticker-chip"
          :style="{ '--chip-color': getTickerColor(index) }"
        >
          <span class="ticker-symbol">{{ ticker.symbol }}</span>
          <span class="ticker-name">{{ ticker.name }}</span>
          <button 
            @click="removeTicker(index)"
            class="remove-btn"
            :aria-label="`Remove ${ticker.symbol}`"
          >
            ×
          </button>
        </div>
      </div>

      <div class="add-ticker-section">
        <div class="ticker-search">
          <input
            v-model="searchQuery"
            @input="searchTickers"
            @focus="showSuggestions = true"
            placeholder="Search tickers to compare..."
            class="search-input"
          />
          
          <!-- Search Suggestions -->
          <div v-if="showSuggestions && suggestions.length > 0" class="suggestions-dropdown">
            <div
              v-for="suggestion in suggestions"
              :key="suggestion.ticker"
              @click="addTicker(suggestion)"
              class="suggestion-item"
            >
              <span class="suggestion-symbol">{{ suggestion.ticker }}</span>
              <span class="suggestion-name">{{ suggestion.name }}</span>
              <span class="suggestion-category">{{ suggestion.category }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Comparison Chart -->
    <div v-if="selectedTickers.length > 0" class="comparison-chart-container">
      <div class="chart-controls">
        <div class="time-period-selector">
          <button
            v-for="period in timePeriods"
            :key="period.value"
            @click="selectedPeriod = period.value"
            :class="['period-btn', { active: selectedPeriod === period.value }]"
          >
            {{ period.label }}
          </button>
        </div>

        <div class="chart-type-selector">
          <button
            v-for="type in chartTypes"
            :key="type.value"
            @click="selectedChartType = type.value"
            :class="['chart-type-btn', { active: selectedChartType === type.value }]"
          >
            {{ type.label }}
          </button>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="chart-loading">
        <LoadingSpinner message="Loading comparison data..." />
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="chart-error">
        <ErrorMessage 
          type="data"
          title="Failed to load comparison data"
          :message="error"
          :actions="[{ label: 'Retry', handler: loadComparisonData, variant: 'btn-primary' }]"
        />
      </div>

      <!-- Chart -->
      <div v-else class="comparison-chart">
        <highcharts
          :options="chartOptions"
          :constructor-type="'stockChart'"
          class="chart"
        />
      </div>
    </div>

    <!-- Comparison Statistics -->
    <div v-if="selectedTickers.length > 1 && !loading" class="comparison-stats">
      <h4 class="stats-title">Correlation Analysis</h4>
      <div class="correlation-matrix">
        <div
          v-for="(row, i) in correlationMatrix"
          :key="i"
          class="correlation-row"
        >
          <div class="row-label">{{ selectedTickers[i]?.symbol }}</div>
          <div
            v-for="(value, j) in row"
            :key="j"
            class="correlation-cell"
            :class="getCorrelationClass(value)"
          >
            {{ value !== null ? value.toFixed(3) : '-' }}
          </div>
        </div>
      </div>
    </div>

    <!-- Performance Summary -->
    <div v-if="selectedTickers.length > 0 && performanceData.length > 0" class="performance-summary">
      <h4 class="performance-title">Performance Summary ({{ selectedPeriod }})</h4>
      <div class="performance-table">
        <div class="performance-header">
          <div class="header-cell">Ticker</div>
          <div class="header-cell">Return</div>
          <div class="header-cell">Volatility</div>
          <div class="header-cell">Sharpe Ratio</div>
          <div class="header-cell">Max Drawdown</div>
        </div>
        <div
          v-for="(perf, index) in performanceData"
          :key="perf.symbol"
          class="performance-row"
        >
          <div class="performance-cell ticker-cell">
            <span 
              class="ticker-indicator"
              :style="{ backgroundColor: getTickerColor(index) }"
            ></span>
            {{ perf.symbol }}
          </div>
          <div class="performance-cell" :class="getPerformanceClass(perf.return)">
            {{ formatPercentage(perf.return) }}
          </div>
          <div class="performance-cell">
            {{ formatPercentage(perf.volatility) }}
          </div>
          <div class="performance-cell" :class="getSharpeClass(perf.sharpeRatio)">
            {{ perf.sharpeRatio?.toFixed(2) || '-' }}
          </div>
          <div class="performance-cell negative">
            {{ formatPercentage(perf.maxDrawdown) }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { listIndices, listSectorETFs, getHistoricalData } from '../services/api'
import { useMultipleTickersData } from '../composables/useWebSocket'
import LoadingSpinner from './LoadingSpinner.vue'
import ErrorMessage from './ErrorMessage.vue'

// Reactive data
const selectedTickers = ref([])
const searchQuery = ref('')
const showSuggestions = ref(false)
const suggestions = ref([])
const allTickers = ref([])
const loading = ref(false)
const error = ref(null)
const selectedPeriod = ref('6mo')
const selectedChartType = ref('normalized')
const chartData = ref([])
const correlationMatrix = ref([])
const performanceData = ref([])
const realTimeData = ref({})
const lastRealTimeUpdate = ref(null)

// WebSocket real-time data for selected tickers
const tickerSymbols = computed(() => selectedTickers.value.map(t => t.symbol))
const { 
  tickersData: wsTickersData, 
  isConnected: wsConnected, 
  isLoading: wsLoading 
} = useMultipleTickersData(tickerSymbols)

// Watch for real-time data updates
watch(wsTickersData, (newData) => {
  if (newData && Object.keys(newData).length > 0) {
    realTimeData.value = { ...newData }
    lastRealTimeUpdate.value = new Date()
    console.log('[MarketComparison] Real-time data updated:', Object.keys(newData))
    
    // Update performance data with real-time prices
    updatePerformanceWithRealTime()
  }
}, { deep: true })

// Update performance data with real-time prices
const updatePerformanceWithRealTime = () => {
  if (performanceData.value.length > 0 && Object.keys(realTimeData.value).length > 0) {
    performanceData.value = performanceData.value.map(perf => {
      const rtData = realTimeData.value[perf.symbol]
      if (rtData && rtData.current_price !== undefined) {
        return {
          ...perf,
          currentPrice: rtData.current_price,
          change: rtData.change || perf.change,
          changePercent: rtData.change_percent || perf.changePercent,
          lastUpdate: new Date()
        }
      }
      return perf
    })
  }
}

// Configuration
const timePeriods = [
  { label: '1M', value: '1mo' },
  { label: '3M', value: '3mo' },
  { label: '6M', value: '6mo' },
  { label: '1Y', value: '1y' },
  { label: '2Y', value: '2y' }
]

const chartTypes = [
  { label: 'Normalized', value: 'normalized' },
  { label: 'Absolute', value: 'absolute' },
  { label: 'Percentage', value: 'percentage' }
]

const tickerColors = [
  '#3B82F6', // Blue
  '#EF4444', // Red
  '#10B981', // Green
  '#F59E0B', // Amber
  '#8B5CF6', // Purple
  '#EC4899', // Pink
  '#06B6D4', // Cyan
  '#84CC16'  // Lime
]

// Computed properties
const chartOptions = computed(() => {
  if (!chartData.value.length) return {}

  const series = chartData.value.map((data, index) => ({
    name: selectedTickers.value[index]?.symbol,
    data: data.map(point => [point.timestamp, point.value]),
    color: getTickerColor(index),
    lineWidth: 2,
    marker: {
      enabled: false,
      states: {
        hover: {
          enabled: true,
          radius: 4
        }
      }
    }
  }))

  return {
    chart: {
      type: 'line',
      height: 400,
      backgroundColor: 'transparent',
      style: {
        fontFamily: 'Inter, system-ui, sans-serif'
      }
    },
    title: {
      text: `${selectedChartType.value === 'normalized' ? 'Normalized' : selectedChartType.value === 'percentage' ? 'Percentage' : 'Absolute'} Price Comparison`,
      style: {
        fontSize: '16px',
        fontWeight: '600'
      }
    },
    xAxis: {
      type: 'datetime',
      gridLineWidth: 1,
      gridLineColor: 'rgba(0,0,0,0.1)'
    },
    yAxis: {
      title: {
        text: selectedChartType.value === 'normalized' ? 'Normalized Value' : 
              selectedChartType.value === 'percentage' ? 'Change (%)' : 'Price ($)'
      },
      gridLineWidth: 1,
      gridLineColor: 'rgba(0,0,0,0.1)'
    },
    legend: {
      enabled: true,
      align: 'center',
      verticalAlign: 'bottom'
    },
    tooltip: {
      shared: true,
      crosshairs: true,
      formatter: function() {
        let tooltip = `<b>${new Date(this.x).toLocaleDateString()}</b><br/>`
        this.points.forEach(point => {
          const value = selectedChartType.value === 'percentage' ? 
            `${point.y.toFixed(2)}%` : 
            selectedChartType.value === 'normalized' ?
            point.y.toFixed(4) :
            `$${point.y.toFixed(2)}`
          tooltip += `<span style="color:${point.color}">●</span> ${point.series.name}: <b>${value}</b><br/>`
        })
        return tooltip
      }
    },
    plotOptions: {
      line: {
        animation: {
          duration: 1000
        }
      }
    },
    series
  }
})

// Methods
function getTickerColor(index) {
  return tickerColors[index % tickerColors.length]
}

async function loadAllTickers() {
  try {
    const [indices, etfs] = await Promise.all([
      listIndices(),
      listSectorETFs()
    ])
    
    allTickers.value = [
      ...indices.map(t => ({ ...t, category: 'index' })),
      ...etfs.map(t => ({ ...t, category: 'etf' }))
    ]
  } catch (err) {
    console.error('Failed to load tickers:', err)
  }
}

function searchTickers() {
  if (!searchQuery.value.trim()) {
    suggestions.value = []
    return
  }

  const query = searchQuery.value.toLowerCase()
  suggestions.value = allTickers.value
    .filter(ticker => 
      ticker.ticker.toLowerCase().includes(query) ||
      ticker.name.toLowerCase().includes(query)
    )
    .slice(0, 8)
}

function addTicker(ticker) {
  if (selectedTickers.value.length >= 8) {
    alert('Maximum 8 tickers can be compared')
    return
  }

  if (selectedTickers.value.some(t => t.symbol === ticker.ticker)) {
    alert('Ticker already selected')
    return
  }

  selectedTickers.value.push({
    symbol: ticker.ticker,
    name: ticker.name,
    category: ticker.category
  })

  searchQuery.value = ''
  showSuggestions.value = false
  loadComparisonData()
}

function removeTicker(index) {
  selectedTickers.value.splice(index, 1)
  if (selectedTickers.value.length > 0) {
    loadComparisonData()
  } else {
    chartData.value = []
    correlationMatrix.value = []
    performanceData.value = []
  }
}

async function loadComparisonData() {
  if (selectedTickers.value.length === 0) return

  loading.value = true
  error.value = null

  try {
    const promises = selectedTickers.value.map(ticker =>
      getHistoricalData(ticker.symbol, ticker.category, {
        period: selectedPeriod.value,
        interval: '1d'
      })
    )

    // Normalize API results to arrays of OHLC objects
    const rawResults = await Promise.all(promises)
    const results = rawResults.map(r => Array.isArray(r) ? r : (r?.data ?? []))
    // Remove invalid points without close/date to prevent runtime errors
    const cleanResults = results.map(series =>
      (series || []).filter(p => p && typeof p.close === 'number' && p.date)
    )

    processChartData(cleanResults)
    calculateCorrelations(cleanResults)
    calculatePerformanceMetrics(cleanResults)
  } catch (err) {
    error.value = err.message || 'Failed to load comparison data'
  } finally {
    loading.value = false
  }
}

function processChartData(results) {
  chartData.value = results.map((data, tickerIndex) => {
    if (!data || !data.length) return []

    const prices = data.map(d => d.close)
    const basePrice = prices[0]

    return data.map((point, index) => {
      let value
      if (selectedChartType.value === 'normalized') {
        value = point.close / basePrice
      } else if (selectedChartType.value === 'percentage') {
        value = ((point.close - basePrice) / basePrice) * 100
      } else {
        value = point.close
      }

      return {
        timestamp: new Date(point.date).getTime(),
        value
      }
    })
  })
}

function calculateCorrelations(results) {
  const n = results.length
  if (n < 2) {
    correlationMatrix.value = []
    return
  }

  const matrix = Array(n).fill().map(() => Array(n).fill(null))
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1.0
      } else if (results[i] && results[j] && results[i].length > 0 && results[j].length > 0) {
        const returns1 = calculateReturns(results[i])
        const returns2 = calculateReturns(results[j])
        matrix[i][j] = calculateCorrelation(returns1, returns2)
      }
    }
  }

  correlationMatrix.value = matrix
}

function calculateReturns(data) {
  const returns = []
  for (let i = 1; i < data.length; i++) {
    const prevPrice = data[i - 1].close
    const currentPrice = data[i].close
    returns.push((currentPrice - prevPrice) / prevPrice)
  }
  return returns
}

function calculateCorrelation(returns1, returns2) {
  if (returns1.length !== returns2.length || returns1.length === 0) return null

  const n = returns1.length
  const mean1 = returns1.reduce((sum, r) => sum + r, 0) / n
  const mean2 = returns2.reduce((sum, r) => sum + r, 0) / n

  let numerator = 0
  let sum1Sq = 0
  let sum2Sq = 0

  for (let i = 0; i < n; i++) {
    const diff1 = returns1[i] - mean1
    const diff2 = returns2[i] - mean2
    numerator += diff1 * diff2
    sum1Sq += diff1 * diff1
    sum2Sq += diff2 * diff2
  }

  const denominator = Math.sqrt(sum1Sq * sum2Sq)
  return denominator === 0 ? null : numerator / denominator
}

function calculatePerformanceMetrics(results) {
  performanceData.value = results.map((data, index) => {
    if (!data || data.length < 2) {
      return {
        symbol: selectedTickers.value[index]?.symbol,
        return: null,
        volatility: null,
        sharpeRatio: null,
        maxDrawdown: null
      }
    }

    const returns = calculateReturns(data)
    const totalReturn = ((data[data.length - 1].close - data[0].close) / data[0].close) * 100
    const volatility = calculateVolatility(returns) * 100
    const sharpeRatio = calculateSharpeRatio(returns)
    const maxDrawdown = calculateMaxDrawdown(data) * 100

    return {
      symbol: selectedTickers.value[index]?.symbol,
      return: totalReturn,
      volatility,
      sharpeRatio,
      maxDrawdown
    }
  })
}

function calculateVolatility(returns) {
  if (returns.length === 0) return null
  
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
  return Math.sqrt(variance * 252) // Annualized
}

function calculateSharpeRatio(returns) {
  if (returns.length === 0) return null
  
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
  const volatility = calculateVolatility(returns)
  return volatility === 0 ? null : (mean * 252) / volatility // Annualized
}

function calculateMaxDrawdown(data) {
  let maxDrawdown = 0
  let peak = data[0].close

  for (let i = 1; i < data.length; i++) {
    if (data[i].close > peak) {
      peak = data[i].close
    } else {
      const drawdown = (peak - data[i].close) / peak
      maxDrawdown = Math.max(maxDrawdown, drawdown)
    }
  }

  return maxDrawdown
}

function getCorrelationClass(value) {
  if (value === null) return 'correlation-null'
  if (value > 0.7) return 'correlation-high'
  if (value > 0.3) return 'correlation-medium'
  if (value > -0.3) return 'correlation-low'
  return 'correlation-negative'
}

function getPerformanceClass(value) {
  if (value === null) return ''
  return value >= 0 ? 'positive' : 'negative'
}

function getSharpeClass(value) {
  if (value === null) return ''
  if (value > 1) return 'positive'
  if (value > 0) return 'neutral'
  return 'negative'
}

function formatPercentage(value) {
  if (value === null) return '-'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

// Watchers
watch([selectedPeriod, selectedChartType], () => {
  if (selectedTickers.value.length > 0) {
    loadComparisonData()
  }
})

// Lifecycle
onMounted(() => {
  loadAllTickers()
  
  // Close suggestions when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.ticker-search')) {
      showSuggestions.value = false
    }
  })
})
</script>

<style scoped>
.market-comparison {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
}

.comparison-header {
  text-align: center;
  margin-bottom: var(--space-6);
}

.comparison-title {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  margin-bottom: var(--space-2);
  color: var(--color-text-primary);
}

.comparison-subtitle {
  color: var(--color-text-secondary);
  margin: 0;
}

/* Ticker Selection */
.ticker-selection {
  margin-bottom: var(--space-6);
}

.selected-tickers {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.ticker-chip {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background: var(--color-surface-elevated);
  border: 2px solid var(--chip-color);
  border-radius: var(--radius-full);
  padding: var(--space-2) var(--space-3);
  font-size: var(--text-sm);
}

.ticker-symbol {
  font-weight: var(--font-bold);
  color: var(--chip-color);
}

.ticker-name {
  color: var(--color-text-secondary);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.remove-btn {
  background: none;
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: var(--text-lg);
  line-height: 1;
  padding: 0;
  margin-left: var(--space-1);
}

.remove-btn:hover {
  color: var(--color-danger);
}

/* Search */
.ticker-search {
  position: relative;
  max-width: 400px;
}

.search-input {
  width: 100%;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--text-base);
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.search-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

.suggestions-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  z-index: 10;
  max-height: 300px;
  overflow-y: auto;
}

.suggestion-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  cursor: pointer;
  border-bottom: 1px solid var(--color-border);
}

.suggestion-item:hover {
  background: var(--color-surface-elevated);
}

.suggestion-item:last-child {
  border-bottom: none;
}

.suggestion-symbol {
  font-weight: var(--font-bold);
  color: var(--color-primary);
  min-width: 60px;
}

.suggestion-name {
  flex: 1;
  color: var(--color-text-primary);
}

.suggestion-category {
  font-size: var(--text-xs);
  color: var(--color-text-secondary);
  text-transform: uppercase;
  background: var(--color-surface-elevated);
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
}

/* Chart Controls */
.chart-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-4);
  flex-wrap: wrap;
  gap: var(--space-4);
}

.time-period-selector,
.chart-type-selector {
  display: flex;
  gap: var(--space-2);
}

.period-btn,
.chart-type-btn {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text-secondary);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--text-sm);
  transition: all var(--transition-fast);
}

.period-btn:hover,
.chart-type-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-text-primary);
}

.period-btn.active,
.chart-type-btn.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

/* Chart */
.comparison-chart-container {
  margin-bottom: var(--space-6);
}

.chart-loading,
.chart-error {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.comparison-chart {
  background: var(--color-surface);
  border-radius: var(--radius-md);
  overflow: hidden;
}

/* Statistics */
.comparison-stats {
  margin-bottom: var(--space-6);
}

.stats-title {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  margin-bottom: var(--space-4);
  color: var(--color-text-primary);
}

.correlation-matrix {
  display: grid;
  gap: var(--space-1);
  font-size: var(--text-sm);
}

.correlation-row {
  display: grid;
  grid-template-columns: 80px repeat(auto-fit, minmax(60px, 1fr));
  gap: var(--space-1);
  align-items: center;
}

.row-label {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
  padding: var(--space-2);
}

.correlation-cell {
  padding: var(--space-2);
  text-align: center;
  border-radius: var(--radius-sm);
  font-weight: var(--font-medium);
}

.correlation-high {
  background: var(--color-success-light);
  color: var(--color-success);
}

.correlation-medium {
  background: var(--color-warning-light);
  color: var(--color-warning);
}

.correlation-low {
  background: var(--color-surface-elevated);
  color: var(--color-text-secondary);
}

.correlation-negative {
  background: var(--color-danger-light);
  color: var(--color-danger);
}

.correlation-null {
  background: var(--color-surface-elevated);
  color: var(--color-text-secondary);
}

/* Performance Table */
.performance-summary {
  margin-top: var(--space-6);
}

.performance-title {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  margin-bottom: var(--space-4);
  color: var(--color-text-primary);
}

.performance-table {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.performance-header {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
  background: var(--color-surface-elevated);
  border-bottom: 1px solid var(--color-border);
}

.header-cell {
  padding: var(--space-3) var(--space-4);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  text-align: center;
}

.performance-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
  border-bottom: 1px solid var(--color-border);
}

.performance-row:last-child {
  border-bottom: none;
}

.performance-cell {
  padding: var(--space-3) var(--space-4);
  text-align: center;
  color: var(--color-text-primary);
}

.ticker-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  font-weight: var(--font-medium);
}

.ticker-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.positive {
  color: var(--color-success);
}

.negative {
  color: var(--color-danger);
}

.neutral {
  color: var(--color-warning);
}

/* Responsive Design */
@media (max-width: 768px) {
  .chart-controls {
    flex-direction: column;
    align-items: stretch;
  }

  .time-period-selector,
  .chart-type-selector {
    justify-content: center;
  }

  .correlation-row {
    grid-template-columns: 60px repeat(auto-fit, minmax(50px, 1fr));
  }

  .performance-header,
  .performance-row {
    grid-template-columns: 1fr;
    text-align: left;
  }

  .performance-cell {
    text-align: left;
    border-bottom: 1px solid var(--color-border);
  }

  .performance-cell:last-child {
    border-bottom: none;
  }
}
</style>
