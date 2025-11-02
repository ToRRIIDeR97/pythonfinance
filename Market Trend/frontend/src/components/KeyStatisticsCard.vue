<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { getSummary } from '../services/api'
import { useRoute } from 'vue-router'
import { useMarketStore } from '../stores/market'

const route = useRoute()
const store = useMarketStore()
const ticker = computed(() => route.params.ticker || store.selectedTicker)
const loading = ref(true)
const error = ref(null)
const stats = ref(null)

async function loadStats() {
  if (!ticker.value) return
  loading.value = true
  error.value = null
  try {
    // Always use 1y period for 52W metrics, 1d interval
    stats.value = await getSummary(ticker.value, { period: '1y', interval: '1d' })
  } catch (e) {
    error.value = e?.message || 'Failed to load key statistics'
  } finally {
    loading.value = false
  }
}

onMounted(loadStats)
watch(() => ticker.value, loadStats)
</script>

<template>
  <div class="card">
    <div class="card-title">Key Statistics</div>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <div v-if="error" class="error">{{ error }}</div>
      <div v-else-if="!stats" class="empty">No data</div>
      <div v-else class="grid">
        <div class="cell">
          <div class="label">Price</div>
          <div class="value">
            <span>{{ stats.price?.toFixed?.(2) ?? stats.price }}</span>
            <span :class="['delta', stats.change > 0 ? 'up' : stats.change < 0 ? 'down' : '']">
              {{ stats.change > 0 ? '+' : '' }}{{ stats.change?.toFixed?.(2) ?? stats.change }}
              ({{ stats.change_pct?.toFixed?.(2) ?? stats.change_pct }}%)
            </span>
          </div>
        </div>
        <div class="cell">
          <div class="label">52W High</div>
          <div class="value">{{ stats.high_52w?.toFixed?.(2) ?? stats.high_52w }}</div>
        </div>
        <div class="cell">
          <div class="label">52W Low</div>
          <div class="value">{{ stats.low_52w?.toFixed?.(2) ?? stats.low_52w }}</div>
        </div>
        <div class="cell">
          <div class="label">SMA 20</div>
          <div class="value">{{ stats.sma_20?.toFixed?.(2) ?? stats.sma_20 }}</div>
        </div>
        <div class="cell">
          <div class="label">SMA 50</div>
          <div class="value">{{ stats.sma_50?.toFixed?.(2) ?? stats.sma_50 }}</div>
        </div>
        <div class="cell">
          <div class="label">SMA 200</div>
          <div class="value">{{ stats.sma_200?.toFixed?.(2) ?? stats.sma_200 }}</div>
        </div>
        <div class="cell">
          <div class="label">RSI 14</div>
          <div class="value">{{ stats.rsi_14?.toFixed?.(1) ?? stats.rsi_14 }}</div>
        </div>
        <div class="cell">
          <div class="label">Avg Vol (20d)</div>
          <div class="value">{{ stats.volume_avg_20d?.toLocaleString?.() ?? stats.volume_avg_20d }}</div>
        </div>
      </div>
    </div>
  </div>
  </template>

<style scoped>
.card { border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.card-title { font-weight: 600; margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.cell { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 6px; padding: 8px; }
.label { color: var(--text-color); opacity: 0.7; font-size: 12px; }
.value { font-size: 14px; font-weight: 600; }
.delta { margin-left: 8px; font-size: 12px; }
.up { color: #16a34a; }
.down { color: #dc2626; }
.error { color: #ef4444; }
.empty { color: var(--text-color); opacity: 0.7; }
@media (max-width: 768px) { .grid { grid-template-columns: repeat(2, 1fr); } }
</style>
