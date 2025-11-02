<template>
  <div class="market-heatmap">
    <div class="heatmap-header">
      <h3 class="heatmap-title">Market Heatmap</h3>
      <p class="heatmap-subtitle">Sector performance visualization</p>
    </div>

    <!-- Controls -->
    <div class="heatmap-controls">
      <div class="time-frame-selector">
        <label class="control-label">Time Frame:</label>
        <div class="time-frame-buttons">
          <button
            v-for="timeFrame in timeFrames"
            :key="timeFrame.value"
            @click="selectedTimeFrame = timeFrame.value"
            :class="['time-frame-btn', { active: selectedTimeFrame === timeFrame.value }]"
          >
            {{ timeFrame.label }}
          </button>
        </div>
      </div>

      <div class="metric-selector">
        <label class="control-label">Metric:</label>
        <select v-model="selectedMetric" class="metric-select">
          <option value="return">Return (%)</option>
          <option value="volume">Volume</option>
          <option value="marketCap">Market Cap</option>
          <option value="volatility">Volatility</option>
        </select>
      </div>

      <div class="view-mode-selector">
        <label class="control-label">View:</label>
        <div class="view-mode-buttons">
          <button
            v-for="mode in viewModes"
            :key="mode.value"
            @click="selectedViewMode = mode.value"
            :class="['view-mode-btn', { active: selectedViewMode === mode.value }]"
          >
            {{ mode.label }}
          </button>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="heatmap-loading">
      <LoadingSpinner message="Loading heatmap data..." />
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="heatmap-error">
      <ErrorMessage 
        type="data"
        title="Failed to load heatmap data"
        :message="error"
        :actions="[{ label: 'Retry', handler: loadHeatmapData, variant: 'btn-primary' }]"
      />
    </div>

    <!-- Heatmap -->
    <div v-else class="heatmap-container">
      <!-- Sector Grid View -->
      <div v-if="selectedViewMode === 'sectors'" class="sector-heatmap">
        <div class="heatmap-legend">
          <div class="legend-item">
            <span class="legend-color negative-strong"></span>
            <span class="legend-label">-5%+</span>
          </div>
          <div class="legend-item">
            <span class="legend-color negative-medium"></span>
            <span class="legend-label">-2%</span>
          </div>
          <div class="legend-item">
            <span class="legend-color neutral"></span>
            <span class="legend-label">0%</span>
          </div>
          <div class="legend-item">
            <span class="legend-color positive-medium"></span>
            <span class="legend-label">+2%</span>
          </div>
          <div class="legend-item">
            <span class="legend-color positive-strong"></span>
            <span class="legend-label">+5%+</span>
          </div>
        </div>

        <div class="sector-grid">
          <div
            v-for="sector in sectorData"
            :key="sector.symbol"
            @click="drillDownToSector(sector)"
            class="sector-cell"
            :class="getPerformanceClass(sector.performance)"
            :style="{ 
              '--cell-size': getCellSize(sector),
              '--performance-intensity': getPerformanceIntensity(sector.performance)
            }"
          >
            <div class="sector-info">
              <div class="sector-symbol">{{ sector.symbol }}</div>
              <div class="sector-name">{{ sector.name }}</div>
              <div class="sector-performance">
                {{ formatMetricValue(sector.performance, selectedMetric) }}
              </div>
              <div class="sector-change">
                {{ formatChange(sector.change) }}
              </div>
            </div>
            
            <!-- Mini trend indicator -->
            <div class="trend-indicator">
              <svg width="40" height="20" class="trend-chart">
                <polyline
                  :points="generateTrendPoints(sector.trend)"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.5"
                  opacity="0.7"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      <!-- Individual Stocks View -->
      <div v-else-if="selectedViewMode === 'stocks'" class="stocks-heatmap">
        <div class="breadcrumb" v-if="selectedSector">
          <button @click="backToSectors" class="breadcrumb-btn">
            ← Back to Sectors
          </button>
          <span class="breadcrumb-current">{{ selectedSector.name }}</span>
        </div>

        <div class="stocks-grid">
          <div
            v-for="stock in stockData"
            :key="stock.symbol"
            @click="viewStockDetails(stock)"
            class="stock-cell"
            :class="getPerformanceClass(stock.performance)"
            :style="{ 
              '--performance-intensity': getPerformanceIntensity(stock.performance)
            }"
          >
            <div class="stock-info">
              <div class="stock-symbol">{{ stock.symbol }}</div>
              <div class="stock-name">{{ stock.name }}</div>
              <div class="stock-price">${{ stock.price?.toFixed(2) }}</div>
              <div class="stock-performance">
                {{ formatMetricValue(stock.performance, selectedMetric) }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Treemap View -->
      <div v-else-if="selectedViewMode === 'treemap'" class="treemap-container">
        <div class="treemap-chart">
          <highcharts
            :options="treemapOptions"
            class="treemap"
          />
        </div>
      </div>
    </div>

    <!-- Sector Details Modal -->
    <div v-if="showSectorModal" class="sector-modal-overlay" @click="closeSectorModal">
      <div class="sector-modal" @click.stop>
        <div class="modal-header">
          <h4 class="modal-title">{{ selectedSectorDetails?.name }}</h4>
          <button @click="closeSectorModal" class="modal-close">×</button>
        </div>
        
        <div class="modal-content">
          <div class="sector-stats">
            <div class="stat-item">
              <span class="stat-label">Performance</span>
              <span class="stat-value" :class="getPerformanceClass(selectedSectorDetails?.performance)">
                {{ formatMetricValue(selectedSectorDetails?.performance, 'return') }}
              </span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Market Cap</span>
              <span class="stat-value">{{ formatMarketCap(selectedSectorDetails?.marketCap) }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Volume</span>
              <span class="stat-value">{{ formatVolume(selectedSectorDetails?.volume) }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Volatility</span>
              <span class="stat-value">{{ formatMetricValue(selectedSectorDetails?.volatility, 'volatility') }}</span>
            </div>
          </div>

          <div class="top-performers">
            <h5>Top Performers</h5>
            <div class="performer-list">
              <div
                v-for="performer in selectedSectorDetails?.topPerformers || []"
                :key="performer.symbol"
                class="performer-item"
              >
                <span class="performer-symbol">{{ performer.symbol }}</span>
                <span class="performer-change" :class="getPerformanceClass(performer.change)">
                  {{ formatChange(performer.change) }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { listSectorETFs, getHistoricalData, getSummaryData } from '../services/api'
import { useMultipleTickersData } from '../composables/useWebSocket'
import LoadingSpinner from './LoadingSpinner.vue'
import ErrorMessage from './ErrorMessage.vue'

// Reactive data
const loading = ref(false)
const error = ref(null)
const selectedTimeFrame = ref('1d')
const selectedMetric = ref('return')
const selectedViewMode = ref('sectors')
const selectedSector = ref(null)
const selectedSectorDetails = ref(null)
const showSectorModal = ref(false)
const sectorData = ref([])
const stockData = ref([])
const realTimeData = ref({})
const lastRealTimeUpdate = ref(null)

// WebSocket real-time data for sector ETFs
const sectorSymbols = computed(() => sectorData.value.map(s => s.symbol))
const { 
  tickersData: wsTickersData, 
  isConnected: wsConnected, 
  isLoading: wsLoading 
} = useMultipleTickersData(sectorSymbols)

// Watch for real-time data updates
watch(wsTickersData, (newData) => {
  if (newData && Object.keys(newData).length > 0) {
    realTimeData.value = { ...newData }
    lastRealTimeUpdate.value = new Date()
    console.log('[MarketHeatmap] Real-time data updated:', Object.keys(newData))
    
    // Update sector data with real-time prices
    updateSectorDataWithRealTime()
  }
}, { deep: true })

// Update sector data with real-time prices
const updateSectorDataWithRealTime = () => {
  if (sectorData.value.length > 0 && Object.keys(realTimeData.value).length > 0) {
    sectorData.value = sectorData.value.map(sector => {
      const rtData = realTimeData.value[sector.symbol]
      if (rtData && rtData.current_price !== undefined) {
        return {
          ...sector,
          currentPrice: rtData.current_price,
          change: rtData.change || sector.change,
          performance: rtData.change_percent || sector.performance,
          volume: rtData.volume || sector.volume,
          lastUpdate: new Date()
        }
      }
      return sector
    })
  }
}

// Configuration
const timeFrames = [
  { label: '1D', value: '1d' },
  { label: '1W', value: '1w' },
  { label: '1M', value: '1mo' },
  { label: '3M', value: '3mo' },
  { label: '6M', value: '6mo' },
  { label: '1Y', value: '1y' }
]

const viewModes = [
  { label: 'Sectors', value: 'sectors' },
  { label: 'Stocks', value: 'stocks' },
  { label: 'Treemap', value: 'treemap' }
]

// Mock sector data for demonstration
const mockSectorData = [
  { symbol: 'XLK', name: 'Technology', performance: 2.45, change: 1.23, marketCap: 1200000000000, volume: 45000000, volatility: 18.5 },
  { symbol: 'XLF', name: 'Financial', performance: -0.87, change: -0.45, marketCap: 800000000000, volume: 32000000, volatility: 22.1 },
  { symbol: 'XLV', name: 'Healthcare', performance: 1.12, change: 0.67, marketCap: 950000000000, volume: 28000000, volatility: 15.8 },
  { symbol: 'XLE', name: 'Energy', performance: -3.21, change: -2.15, marketCap: 450000000000, volume: 55000000, volatility: 35.2 },
  { symbol: 'XLI', name: 'Industrial', performance: 0.95, change: 0.34, marketCap: 650000000000, volume: 22000000, volatility: 19.7 },
  { symbol: 'XLY', name: 'Consumer Disc.', performance: 1.78, change: 0.89, marketCap: 720000000000, volume: 35000000, volatility: 21.3 },
  { symbol: 'XLP', name: 'Consumer Staples', performance: -0.23, change: -0.12, marketCap: 580000000000, volume: 18000000, volatility: 12.4 },
  { symbol: 'XLU', name: 'Utilities', performance: -1.45, change: -0.78, marketCap: 320000000000, volume: 15000000, volatility: 14.6 },
  { symbol: 'XLB', name: 'Materials', performance: 0.67, change: 0.23, marketCap: 380000000000, volume: 25000000, volatility: 24.8 },
  { symbol: 'XLRE', name: 'Real Estate', performance: -2.10, change: -1.34, marketCap: 280000000000, volume: 12000000, volatility: 20.9 }
]

// Computed properties
const treemapOptions = computed(() => {
  if (!sectorData.value.length) return {}

  const data = sectorData.value.map(sector => ({
    name: sector.name,
    value: Math.abs(sector.marketCap || 1000000000),
    colorValue: sector.performance,
    symbol: sector.symbol
  }))

  return {
    chart: {
      type: 'treemap',
      height: 500,
      backgroundColor: 'transparent'
    },
    title: {
      text: `Market Treemap - ${selectedMetric.value === 'return' ? 'Performance' : selectedMetric.value}`,
      style: {
        fontSize: '16px',
        fontWeight: '600'
      }
    },
    colorAxis: {
      minColor: '#ef4444',
      maxColor: '#10b981',
      stops: [
        [0, '#ef4444'],
        [0.5, '#f3f4f6'],
        [1, '#10b981']
      ]
    },
    series: [{
      type: 'treemap',
      layoutAlgorithm: 'squarified',
      data: data,
      dataLabels: {
        enabled: true,
        format: '{point.name}<br/>{point.colorValue:.2f}%',
        style: {
          fontWeight: 'bold',
          fontSize: '12px'
        }
      },
      tooltip: {
        pointFormat: '<b>{point.name}</b><br/>Performance: {point.colorValue:.2f}%<br/>Market Cap: ${point.value:,.0f}'
      }
    }]
  }
})

// Methods
async function loadHeatmapData() {
  loading.value = true
  error.value = null

  try {
    // For now, use mock data. In production, this would fetch real sector data
    sectorData.value = mockSectorData.map(sector => ({
      ...sector,
      trend: generateMockTrend()
    }))
  } catch (err) {
    error.value = err.message || 'Failed to load heatmap data'
  } finally {
    loading.value = false
  }
}

function generateMockTrend() {
  const points = []
  let value = 50
  for (let i = 0; i < 10; i++) {
    value += (Math.random() - 0.5) * 10
    points.push(Math.max(10, Math.min(90, value)))
  }
  return points
}

function generateTrendPoints(trend) {
  if (!trend || !trend.length) return ''
  
  const width = 40
  const height = 20
  const stepX = width / (trend.length - 1)
  
  return trend.map((value, index) => {
    const x = index * stepX
    const y = height - (value / 100) * height
    return `${x},${y}`
  }).join(' ')
}

function getPerformanceClass(performance) {
  if (performance === null || performance === undefined) return 'neutral'
  if (performance > 2) return 'positive-strong'
  if (performance > 0.5) return 'positive-medium'
  if (performance > -0.5) return 'neutral'
  if (performance > -2) return 'negative-medium'
  return 'negative-strong'
}

function getPerformanceIntensity(performance) {
  if (performance === null || performance === undefined) return 0.1
  return Math.min(1, Math.abs(performance) / 5)
}

function getCellSize(sector) {
  if (selectedMetric.value === 'marketCap') {
    const maxMarketCap = Math.max(...sectorData.value.map(s => s.marketCap || 0))
    const minSize = 120
    const maxSize = 200
    const ratio = (sector.marketCap || 0) / maxMarketCap
    return `${minSize + (maxSize - minSize) * ratio}px`
  }
  return '150px'
}

function formatMetricValue(value, metric) {
  if (value === null || value === undefined) return '-'
  
  switch (metric) {
    case 'return':
      return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
    case 'volume':
      return formatVolume(value)
    case 'marketCap':
      return formatMarketCap(value)
    case 'volatility':
      return `${value.toFixed(1)}%`
    default:
      return value.toString()
  }
}

function formatChange(change) {
  if (change === null || change === undefined) return '-'
  return `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`
}

function formatVolume(volume) {
  if (!volume) return '-'
  if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`
  if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`
  return volume.toString()
}

function formatMarketCap(marketCap) {
  if (!marketCap) return '-'
  if (marketCap >= 1000000000000) return `$${(marketCap / 1000000000000).toFixed(1)}T`
  if (marketCap >= 1000000000) return `$${(marketCap / 1000000000).toFixed(1)}B`
  if (marketCap >= 1000000) return `$${(marketCap / 1000000).toFixed(1)}M`
  return `$${marketCap.toFixed(0)}`
}

function drillDownToSector(sector) {
  selectedSector.value = sector
  selectedSectorDetails.value = {
    ...sector,
    topPerformers: [
      { symbol: 'AAPL', change: 2.45 },
      { symbol: 'MSFT', change: 1.87 },
      { symbol: 'GOOGL', change: 1.23 }
    ]
  }
  showSectorModal.value = true
}

function closeSectorModal() {
  showSectorModal.value = false
  selectedSectorDetails.value = null
}

function backToSectors() {
  selectedSector.value = null
  selectedViewMode.value = 'sectors'
}

function viewStockDetails(stock) {
  // Navigate to individual stock chart
  console.log('View stock details:', stock.symbol)
}

// Watchers
watch([selectedTimeFrame, selectedMetric], () => {
  loadHeatmapData()
})

// Lifecycle
onMounted(() => {
  loadHeatmapData()
})
</script>

<style scoped>
.market-heatmap {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
}

.heatmap-header {
  text-align: center;
  margin-bottom: var(--space-6);
}

.heatmap-title {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  margin-bottom: var(--space-2);
  color: var(--color-text-primary);
}

.heatmap-subtitle {
  color: var(--color-text-secondary);
  margin: 0;
}

/* Controls */
.heatmap-controls {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-6);
  align-items: center;
  justify-content: center;
  margin-bottom: var(--space-6);
  padding: var(--space-4);
  background: var(--color-surface-elevated);
  border-radius: var(--radius-md);
}

.control-label {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
  margin-right: var(--space-2);
}

.time-frame-buttons,
.view-mode-buttons {
  display: flex;
  gap: var(--space-2);
}

.time-frame-btn,
.view-mode-btn {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text-secondary);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--text-sm);
  transition: all var(--transition-fast);
}

.time-frame-btn:hover,
.view-mode-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-text-primary);
}

.time-frame-btn.active,
.view-mode-btn.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.metric-select {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  color: var(--color-text-primary);
  font-size: var(--text-sm);
}

.metric-select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

/* Loading and Error States */
.heatmap-loading,
.heatmap-error {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

/* Legend */
.heatmap-legend {
  display: flex;
  justify-content: center;
  gap: var(--space-4);
  margin-bottom: var(--space-4);
  padding: var(--space-3);
  background: var(--color-surface-elevated);
  border-radius: var(--radius-md);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: var(--radius-sm);
}

.legend-label {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
}

/* Sector Grid */
.sector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-6);
}

.sector-cell {
  width: var(--cell-size, 150px);
  height: var(--cell-size, 150px);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  cursor: pointer;
  transition: all var(--transition-fast);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
  border: 2px solid transparent;
}

.sector-cell:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
  border-color: var(--color-primary);
}

.sector-info {
  z-index: 2;
}

.sector-symbol {
  font-weight: var(--font-bold);
  font-size: var(--text-lg);
  color: var(--color-text-primary);
  margin-bottom: var(--space-1);
}

.sector-name {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin-bottom: var(--space-2);
  line-height: var(--leading-tight);
}

.sector-performance {
  font-weight: var(--font-bold);
  font-size: var(--text-xl);
  margin-bottom: var(--space-1);
}

.sector-change {
  font-size: var(--text-sm);
  opacity: 0.8;
}

.trend-indicator {
  position: absolute;
  bottom: var(--space-2);
  right: var(--space-2);
  opacity: 0.6;
}

/* Performance Colors */
.positive-strong {
  background: linear-gradient(135deg, rgba(16, 185, 129, var(--performance-intensity, 0.8)) 0%, rgba(16, 185, 129, 0.6) 100%);
  color: white;
}

.positive-medium {
  background: linear-gradient(135deg, rgba(34, 197, 94, var(--performance-intensity, 0.6)) 0%, rgba(34, 197, 94, 0.4) 100%);
  color: var(--color-text-primary);
}

.neutral {
  background: linear-gradient(135deg, rgba(107, 114, 128, 0.3) 0%, rgba(107, 114, 128, 0.1) 100%);
  color: var(--color-text-primary);
}

.negative-medium {
  background: linear-gradient(135deg, rgba(239, 68, 68, var(--performance-intensity, 0.6)) 0%, rgba(239, 68, 68, 0.4) 100%);
  color: var(--color-text-primary);
}

.negative-strong {
  background: linear-gradient(135deg, rgba(220, 38, 38, var(--performance-intensity, 0.8)) 0%, rgba(220, 38, 38, 0.6) 100%);
  color: white;
}

.legend-color.positive-strong { background: #10b981; }
.legend-color.positive-medium { background: #22c55e; }
.legend-color.neutral { background: #6b7280; }
.legend-color.negative-medium { background: #ef4444; }
.legend-color.negative-strong { background: #dc2626; }

/* Stocks Grid */
.stocks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--space-3);
}

.stock-cell {
  padding: var(--space-4);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
  border: 1px solid var(--color-border);
}

.stock-cell:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.stock-symbol {
  font-weight: var(--font-bold);
  font-size: var(--text-base);
  margin-bottom: var(--space-1);
}

.stock-name {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin-bottom: var(--space-2);
}

.stock-price {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  margin-bottom: var(--space-1);
}

.stock-performance {
  font-weight: var(--font-medium);
}

/* Breadcrumb */
.breadcrumb {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-4);
  padding: var(--space-3);
  background: var(--color-surface-elevated);
  border-radius: var(--radius-md);
}

.breadcrumb-btn {
  background: none;
  border: none;
  color: var(--color-primary);
  cursor: pointer;
  font-size: var(--text-sm);
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
}

.breadcrumb-btn:hover {
  background: var(--color-primary-light);
}

.breadcrumb-current {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
}

/* Treemap */
.treemap-container {
  min-height: 500px;
}

.treemap {
  width: 100%;
  height: 500px;
}

/* Modal */
.sector-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.sector-modal {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  max-width: 500px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: var(--shadow-xl);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-4);
  padding-bottom: var(--space-3);
  border-bottom: 1px solid var(--color-border);
}

.modal-title {
  font-size: var(--text-xl);
  font-weight: var(--font-bold);
  color: var(--color-text-primary);
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: var(--text-2xl);
  color: var(--color-text-secondary);
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.modal-close:hover {
  color: var(--color-text-primary);
}

.sector-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-4);
  margin-bottom: var(--space-6);
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.stat-label {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
}

.stat-value {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
}

.top-performers h5 {
  font-size: var(--text-base);
  font-weight: var(--font-semibold);
  margin-bottom: var(--space-3);
  color: var(--color-text-primary);
}

.performer-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.performer-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-2) var(--space-3);
  background: var(--color-surface-elevated);
  border-radius: var(--radius-md);
}

.performer-symbol {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
}

.performer-change {
  font-weight: var(--font-semibold);
}

/* Performance text colors */
.positive-strong,
.positive-medium {
  color: var(--color-success);
}

.negative-strong,
.negative-medium {
  color: var(--color-danger);
}

/* Responsive Design */
@media (max-width: 768px) {
  .heatmap-controls {
    flex-direction: column;
    gap: var(--space-4);
  }

  .sector-grid {
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  }

  .sector-cell {
    width: 120px;
    height: 120px;
    padding: var(--space-3);
  }

  .stocks-grid {
    grid-template-columns: 1fr;
  }

  .sector-stats {
    grid-template-columns: 1fr;
  }
}
</style>