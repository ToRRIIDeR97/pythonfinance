<script setup>
import { ref, computed, onMounted } from 'vue'
import { VueDraggableNext } from 'vue-draggable-next'
import { listIndices, listSectorETFs } from '../services/api'
import { useRouter } from 'vue-router'
import { useMarketStore } from '../stores/market'
import MarketCard from '../components/MarketCard.vue'
import MarketComparison from '../components/MarketComparison.vue'
import MarketHeatmap from '../components/MarketHeatmap.vue'
import LoadingSpinner from '../components/LoadingSpinner.vue'
import ErrorMessage from '../components/ErrorMessage.vue'
import SearchFilter from '../components/SearchFilter.vue'

const indices = ref([])
const etfs = ref([])
const loading = ref(true)
const error = ref(null)
const router = useRouter()
const market = useMarketStore()
const customizeMode = ref(false)

// Filter state
const filteredData = ref([])
const searchQuery = ref('')

// Combined data for filtering
const allMarketData = computed(() => [
  ...indices.value,
  ...etfs.value
])

// Filtered indices and ETFs
const filteredIndices = computed(() => {
  if (!searchQuery.value && filteredData.value.length === 0) {
    return indices.value
  }
  return filteredData.value.filter(item => item.category === 'index')
})

const filteredETFs = computed(() => {
  if (!searchQuery.value && filteredData.value.length === 0) {
    return etfs.value
  }
  return filteredData.value.filter(item => item.category === 'etf')
})

// Filter handlers
const handleFiltered = (data) => {
  filteredData.value = data
}

const handleSearch = (query) => {
  searchQuery.value = query
}

const canCustomize = computed(() => !loading.value && !error.value && !searchQuery.value)

function applyLayout(list, order) {
  if (!order || order.length === 0) return list
  const byTicker = new Map(list.map(item => [item.ticker, item]))
  const inOrder = order.map(t => byTicker.get(t)).filter(Boolean)
  const missing = list.filter(item => !order.includes(item.ticker))
  return [...inOrder, ...missing]
}

function saveLayout() {
  try {
    localStorage.setItem('indicesLayout', JSON.stringify(indices.value.map(i => i.ticker)))
    localStorage.setItem('etfsLayout', JSON.stringify(etfs.value.map(e => e.ticker)))
  } catch {}
}

function resetLayout() {
  try {
    localStorage.removeItem('indicesLayout')
    localStorage.removeItem('etfsLayout')
    customizeMode.value = false
  } catch {}
}

onMounted(async () => {
  try {
    const [idx, sec] = await Promise.all([listIndices(), listSectorETFs()])
    indices.value = idx.map(i => ({ ...i, category: 'index' }))
    etfs.value = sec.map(s => ({ ...s, category: 'etf' }))
    // apply saved layout if present
    try {
      const savedIdx = JSON.parse(localStorage.getItem('indicesLayout') || '[]')
      const savedEtf = JSON.parse(localStorage.getItem('etfsLayout') || '[]')
      if (savedIdx?.length) indices.value = applyLayout(indices.value, savedIdx)
      if (savedEtf?.length) etfs.value = applyLayout(etfs.value, savedEtf)
    } catch {}
  } catch (e) {
    error.value = e?.message || 'Failed to load tickers'
  } finally {
    loading.value = false
  }
})

function viewChart(t) {
  market.setSelection({ category: t.category, ticker: t.ticker })
  router.push(`/chart/${encodeURIComponent(t.ticker)}`)
}
</script>

<template>
  <div class="dashboard">
    <!-- Hero Section -->
    <section class="hero-section animate-fade-in-down">
      <div class="hero-content">
        <h1 class="hero-title">Market Dashboard</h1>
        <p class="hero-subtitle">Real-time market data and insights at your fingertips</p>
      </div>
    </section>

    <!-- Loading State -->
    <LoadingSpinner 
      v-if="loading" 
      message="Loading market data..."
      size="large"
    />

    <!-- Error State -->
    <ErrorMessage 
      v-else-if="error" 
      type="network"
      title="Unable to load market data"
      :message="error"
      :actions="[{ label: 'Try Again', handler: fetchData, variant: 'btn-primary' }]"
    />

    <!-- Dashboard Content -->
    <div v-else class="dashboard-content">
      <!-- Search and Filter Section -->
      <section class="search-filter-section animate-fade-in-up animate-delay-100">
        <SearchFilter 
          :data="allMarketData"
          @filtered="handleFiltered"
          @search="handleSearch"
          placeholder="Search tickers, companies, or sectors..."
        />
        <div class="customize-controls">
          <label class="customize-toggle">
            <input type="checkbox" v-model="customizeMode" :disabled="!canCustomize" />
            <span>Customize Layout</span>
          </label>
          <button class="btn btn-secondary" @click="saveLayout" :disabled="!customizeMode">Save Layout</button>
          <button class="btn btn-outline" @click="resetLayout">Reset</button>
          <small v-if="!canCustomize" class="hint">Disable filters to customize layout</small>
        </div>
      </section>

      <!-- Global Indices Section -->
      <section class="indices-section animate-fade-in-up animate-delay-200">
        <div class="section-header">
          <h2 class="section-title">
            Global Indices
            <span v-if="searchQuery" class="results-count">({{ filteredIndices.length }} results)</span>
          </h2>
          <p class="section-subtitle">Major market indices worldwide</p>
        </div>
        
        <div v-if="!customizeMode && filteredIndices.length > 0" class="indices-grid stagger-children" style="--stagger-delay: 100ms;">
          <MarketCard 
            v-for="(t, index) in filteredIndices" 
            :key="t.ticker"
            :ticker="t.ticker"
            :name="t.name"
            :region="t.region"
            :category="t.category"
            :enable-real-time="true"
            :auto-refresh="true"
            :refresh-interval="30000"
            @click="viewChart(t)"
            class="index-card animate-fade-in-up hover-lift"
            :style="`--stagger-index: ${index}`"
          />
        </div>
        <VueDraggableNext v-else-if="customizeMode" v-model="indices" item-key="ticker" class="indices-grid draggable-grid" :animation="200" ghost-class="drag-ghost" chosen-class="drag-chosen">
          <template #item="{ element, index }">
            <div class="card-wrapper">
              <MarketCard 
                :ticker="element.ticker"
                :name="element.name"
                :region="element.region"
                :category="element.category"
                :enable-real-time="true"
                :auto-refresh="true"
                :refresh-interval="30000"
                @click="viewChart(element)"
                class="index-card"
                :style="`--stagger-index: ${index}`"
              />
            </div>
          </template>
        </VueDraggableNext>
        <div v-else-if="searchQuery" class="no-results">
          <p>No indices found matching "{{ searchQuery }}"</p>
        </div>
      </section>

      <!-- US Sector ETFs Section -->
      <section class="sectors-section animate-fade-in-up animate-delay-300">
        <div class="section-header">
          <h2 class="section-title">
            US Sector ETFs
            <span v-if="searchQuery" class="results-count">({{ filteredETFs.length }} results)</span>
          </h2>
          <p class="section-subtitle">Performance across major sectors</p>
        </div>
        
        <div v-if="!customizeMode && filteredETFs.length > 0" class="sectors-grid stagger-children" style="--stagger-delay: 120ms;">
          <MarketCard 
            v-for="(t, index) in filteredETFs" 
            :key="t.ticker"
            :ticker="t.ticker"
            :name="t.name"
            :category="t.category"
            :enable-real-time="true"
            :auto-refresh="true"
            :refresh-interval="30000"
            @click="viewChart(t)"
            class="sector-card animate-fade-in-up hover-lift"
            :style="`--stagger-index: ${index}`"
          />
        </div>
        <VueDraggableNext v-else-if="customizeMode" v-model="etfs" item-key="ticker" class="sectors-grid draggable-grid" :animation="200" ghost-class="drag-ghost" chosen-class="drag-chosen">
          <template #item="{ element, index }">
            <div class="card-wrapper">
              <MarketCard 
                :ticker="element.ticker"
                :name="element.name"
                :category="element.category"
                :enable-real-time="true"
                :auto-refresh="true"
                :refresh-interval="30000"
                @click="viewChart(element)"
                class="sector-card"
                :style="`--stagger-index: ${index}`"
              />
            </div>
          </template>
        </VueDraggableNext>
        <div v-else-if="searchQuery" class="no-results">
          <p>No ETFs found matching "{{ searchQuery }}"</p>
        </div>
      </section>

      <!-- Market Comparison Section -->
      <section class="comparison-section animate-fade-in-up animate-delay-400">
        <MarketComparison />
      </section>

      <!-- Market Heatmap Section -->
      <section class="heatmap-section animate-fade-in-up animate-delay-500">
        <MarketHeatmap />
      </section>
    </div>
  </div>
</template>

<style scoped>
.dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

.hero-section {
  text-align: center;
  padding: var(--space-8) 0;
  background: linear-gradient(135deg, var(--color-primary-light) 0%, var(--color-surface) 100%);
  border-radius: var(--radius-xl);
  margin-bottom: var(--space-8);
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-16) 0;
  gap: var(--space-4);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--color-border);
  border-top: 3px solid var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-state {
  text-align: center;
  padding: var(--space-16) 0;
}

.error-icon {
  font-size: 3rem;
  margin-bottom: var(--space-4);
}

.section-header {
  text-align: center;
  margin-bottom: var(--space-6);
}

.section-title {
  margin-bottom: var(--space-2);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
}

.results-count {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  font-weight: normal;
}

.section-subtitle {
  color: var(--color-text-secondary);
}

.search-filter-section {
  margin-bottom: var(--space-8);
}

.no-results {
  text-align: center;
  padding: var(--space-12) var(--space-4);
  color: var(--color-text-secondary);
  background: var(--color-background-secondary);
  border-radius: var(--border-radius-lg);
  border: 2px dashed var(--color-border);
}

.no-results p {
  margin: 0;
  font-size: 1.1rem;
}

.indices-grid,
.sectors-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: var(--space-6);
  margin-top: var(--space-6);
}

.index-card,
.sector-card {
  transition: all var(--transition-fast);
}

.index-card:hover,
.sector-card:hover {
  transform: translateY(-2px);
}

.comparison-section {
  margin-top: var(--space-8);
}

.heatmap-section {
  margin-top: var(--space-8);
}

.customize-controls {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-top: var(--space-4);
}

.customize-toggle {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
}

.hint {
  color: var(--color-text-secondary);
}

.draggable-grid {
  /* reuse grid styles */
}

.drag-ghost {
  opacity: 0.6;
}

.drag-chosen {
  transform: scale(0.98);
}

.card-wrapper {
  position: relative;
}

/* Responsive Design */
@media (max-width: 768px) {
  .indices-grid,
  .sectors-grid {
    grid-template-columns: 1fr;
    gap: var(--space-4);
  }
  
  .hero-section {
    padding: var(--space-6) var(--space-4);
    margin-bottom: var(--space-6);
  }
  
  .hero-section h1 {
    font-size: var(--text-3xl);
  }
  
  .section-header h2 {
    font-size: var(--text-2xl);
  }
}

@media (max-width: 480px) {
  .indices-grid,
  .sectors-grid {
    grid-template-columns: 1fr;
    gap: var(--space-3);
  }
}
</style>
