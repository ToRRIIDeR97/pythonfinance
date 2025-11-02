<template>
  <div 
    class="market-card"
    :class="{ 
      'market-card--loading': loading,
      'market-card--positive': changePercent > 0,
      'market-card--negative': changePercent < 0,
      'market-card--clickable': clickable
    }"
    @click="handleClick"
  >
    <!-- Header Section -->
    <div class="market-card__header">
      <div class="market-card__title-section">
        <h3 class="market-card__symbol">{{ ticker }}</h3>
        <p class="market-card__name">{{ name }}</p>
      </div>
      <div class="market-card__region" v-if="region">
        <span class="region-badge">{{ region }}</span>
      </div>
    </div>

    <!-- Price Section -->
    <div class="market-card__price-section">
      <div class="market-card__price-info">
        <div class="current-price">
          <span v-if="loading" class="price-loading">--</span>
          <span v-else class="price-value">${{ formatPrice(currentPrice) }}</span>
        </div>
        <div class="price-change" :class="getChangeClass(changePercent)">
          <span class="change-amount">{{ formatChange(change) }}</span>
          <span class="change-percent">({{ formatPercent(changePercent) }})</span>
        </div>
      </div>
      
      <!-- Volume and Market Cap -->
      <div class="market-card__metrics" v-if="!loading">
        <div class="metric" v-if="volume">
          <span class="metric-label">Vol:</span>
          <span class="metric-value">{{ formatVolume(volume) }}</span>
        </div>
        <div class="metric" v-if="marketCap">
          <span class="metric-label">Cap:</span>
          <span class="metric-value">{{ formatMarketCap(marketCap) }}</span>
        </div>
      </div>
    </div>

    <!-- Mini Chart Section -->
    <div class="market-card__chart-section">
      <MiniChart 
        :ticker="ticker"
        :category="category"
        :height="80"
        :show-tooltip="true"
      />
    </div>

    <!-- Action Section -->
    <div class="market-card__actions">
      <button 
        class="btn btn-primary btn-sm market-card__action-btn"
        @click.stop="$emit('view-chart', { ticker, category, name })"
      >
        <span class="btn-icon">📈</span>
        View Chart
      </button>
      <button 
        class="btn btn-outline btn-sm market-card__action-btn"
        @click.stop="$emit('add-to-watchlist', { ticker, category, name })"
      >
        <span class="btn-icon">⭐</span>
        Watch
      </button>
    </div>

    <!-- Loading Overlay -->
    <div v-if="loading" class="market-card__loading-overlay">
      <div class="loading-spinner"></div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import MiniChart from './MiniChart.vue'
import { getSummaryData } from '../services/api'
import { useTickerData } from '../composables/useWebSocket'

const props = defineProps({
  ticker: {
    type: String,
    required: true
  },
  name: {
    type: String,
    required: true
  },
  category: {
    type: String,
    required: true
  },
  region: {
    type: String,
    default: ''
  },
  clickable: {
    type: Boolean,
    default: true
  },
  autoRefresh: {
    type: Boolean,
    default: false
  },
  refreshInterval: {
    type: Number,
    default: 30000 // 30 seconds
  },
  enableRealTime: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['click', 'view-chart', 'add-to-watchlist'])

// WebSocket real-time data
const { 
  tickerData: realTimeData, 
  isConnected: wsConnected, 
  isLoading: wsLoading 
} = useTickerData(props.ticker)

// Reactive data
const loading = ref(true)
const error = ref(null)
const currentPrice = ref(0)
const change = ref(0)
const changePercent = ref(0)
const volume = ref(0)
const marketCap = ref(0)
const refreshTimer = ref(null)
const lastUpdate = ref(null)
const isRealTimeActive = ref(false)

// Watch for real-time data updates
watch(realTimeData, (newData) => {
  if (newData && props.enableRealTime) {
    updateFromRealTimeData(newData)
    isRealTimeActive.value = true
  }
}, { deep: true })

// Watch WebSocket connection status
watch(wsConnected, (connected) => {
  if (connected && props.enableRealTime) {
    console.log(`[MarketCard] WebSocket connected for ${props.ticker}`)
  } else if (!connected && isRealTimeActive.value) {
    console.log(`[MarketCard] WebSocket disconnected for ${props.ticker}, falling back to polling`)
    // Fall back to polling if WebSocket disconnects
    startPolling()
  }
})

// Update component data from real-time WebSocket data
const updateFromRealTimeData = (data) => {
  if (data.current_price !== undefined) currentPrice.value = data.current_price
  if (data.change !== undefined) change.value = data.change
  if (data.change_percent !== undefined) changePercent.value = data.change_percent
  if (data.volume !== undefined) volume.value = data.volume
  if (data.market_cap !== undefined) marketCap.value = data.market_cap
  
  lastUpdate.value = new Date()
  loading.value = false
  error.value = null
  
  console.log(`[MarketCard] Real-time update for ${props.ticker}:`, {
    price: data.current_price,
    change: data.change_percent
  })
}

// Computed properties
const getChangeClass = (value) => {
  if (value > 0) return 'positive'
  if (value < 0) return 'negative'
  return 'neutral'
}

// Formatting functions
const formatPrice = (price) => {
  if (!price || isNaN(price)) return '0.00'
  return parseFloat(price).toFixed(2)
}

const formatChange = (change) => {
  if (!change || isNaN(change)) return '+0.00'
  const sign = change >= 0 ? '+' : ''
  return `${sign}${parseFloat(change).toFixed(2)}`
}

const formatPercent = (percent) => {
  if (!percent || isNaN(percent)) return '+0.00%'
  const sign = percent >= 0 ? '+' : ''
  return `${sign}${parseFloat(percent).toFixed(2)}%`
}

const formatVolume = (vol) => {
  if (!vol || isNaN(vol)) return '0'
  if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`
  if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`
  if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`
  return vol.toString()
}

const formatMarketCap = (cap) => {
  if (!cap || isNaN(cap)) return 'N/A'
  if (cap >= 1e12) return `${(cap / 1e12).toFixed(1)}T`
  if (cap >= 1e9) return `${(cap / 1e9).toFixed(1)}B`
  if (cap >= 1e6) return `${(cap / 1e6).toFixed(1)}M`
  return cap.toString()
}

// Methods
async function fetchMarketData() {
  if (!props.ticker || !props.category) return
  
  loading.value = true
  error.value = null
  
  try {
    const data = await getSummaryData(props.ticker, props.category)
    if (data) {
      // Only update if not receiving real-time data
      if (!isRealTimeActive.value) {
        // Map backend summary fields with sensible fallbacks
        currentPrice.value = (data.price ?? data.currentPrice ?? data.regularMarketPrice ?? 0)
        change.value = (data.change ?? data.regularMarketChange ?? 0)
        changePercent.value = (data.change_pct ?? data.regularMarketChangePercent ?? 0)
        volume.value = (data.volume ?? data.volume_avg_20d ?? data.regularMarketVolume ?? 0)
        marketCap.value = (data.market_cap ?? data.marketCap ?? 0)
        lastUpdate.value = new Date()
      }
    }
  } catch (err) {
    console.error(`Failed to fetch market data for ${props.ticker}:`, err)
    error.value = err.message || 'Failed to load market data'
  } finally {
    loading.value = false
  }
}

function handleClick() {
  if (props.clickable) {
    emit('click', { ticker: props.ticker, category: props.category, name: props.name })
  }
}

function startPolling() {
  // Start polling when WebSocket is not available
  if (!wsConnected.value && props.autoRefresh && props.refreshInterval > 0) {
    console.log(`[MarketCard] Starting polling for ${props.ticker}`)
    refreshTimer.value = setInterval(fetchMarketData, props.refreshInterval)
  }
}

function startAutoRefresh() {
  // Only start polling if WebSocket is not connected or real-time is disabled
  if ((!wsConnected.value || !props.enableRealTime) && props.autoRefresh && props.refreshInterval > 0) {
    refreshTimer.value = setInterval(fetchMarketData, props.refreshInterval)
  }
}

function stopAutoRefresh() {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
    refreshTimer.value = null
  }
}

// Lifecycle
onMounted(async () => {
  // Initial data fetch
  await fetchMarketData()
  
  // Start auto-refresh if needed (WebSocket will take over if available)
  startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.market-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  position: relative;
  transition: all 0.3s ease;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

.market-card--clickable {
  cursor: pointer;
}

.market-card--clickable:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--color-primary);
}

.market-card--positive {
  border-left: 4px solid var(--color-success);
}

.market-card--negative {
  border-left: 4px solid var(--color-danger);
}

.market-card--loading {
  opacity: 0.7;
}

/* Header Section */
.market-card__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--space-4);
}

.market-card__title-section {
  flex: 1;
}

.market-card__symbol {
  font-size: var(--text-lg);
  font-weight: var(--font-bold);
  color: var(--color-primary);
  margin: 0 0 var(--space-1) 0;
  font-family: var(--font-family-mono);
}

.market-card__name {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin: 0;
  line-height: 1.4;
}

.region-badge {
  background: var(--color-primary);
  color: white;
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
}

/* Price Section */
.market-card__price-section {
  margin-bottom: var(--space-4);
}

.market-card__price-info {
  margin-bottom: var(--space-3);
}

.current-price {
  margin-bottom: var(--space-1);
}

.price-value {
  font-size: var(--text-xl);
  font-weight: var(--font-bold);
  color: var(--color-text-primary);
  font-family: var(--font-family-mono);
}

.price-loading {
  font-size: var(--text-xl);
  color: var(--color-text-secondary);
  font-family: var(--font-family-mono);
}

.price-change {
  display: flex;
  gap: var(--space-2);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  font-family: var(--font-family-mono);
}

.positive {
  color: var(--color-success);
}

.negative {
  color: var(--color-danger);
}

.neutral {
  color: var(--color-text-secondary);
}

/* Metrics */
.market-card__metrics {
  display: flex;
  gap: var(--space-4);
}

.metric {
  display: flex;
  gap: var(--space-1);
  font-size: var(--text-xs);
}

.metric-label {
  color: var(--color-text-secondary);
  font-weight: var(--font-medium);
}

.metric-value {
  color: var(--color-text-primary);
  font-family: var(--font-family-mono);
}

/* Chart Section */
.market-card__chart-section {
  margin-bottom: var(--space-4);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--color-surface-elevated);
}

/* Actions */
.market-card__actions {
  display: flex;
  gap: var(--space-2);
}

.market-card__action-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-1);
}

.btn-icon {
  font-size: var(--text-sm);
}

/* Loading Overlay */
.market-card__loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}

.dark .market-card__loading-overlay {
  background: rgba(0, 0, 0, 0.8);
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--color-border);
  border-top: 2px solid var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
  .market-card {
    padding: var(--space-4);
  }
  
  .market-card__actions {
    flex-direction: column;
  }
  
  .market-card__metrics {
    flex-direction: column;
    gap: var(--space-2);
  }
}
</style>
