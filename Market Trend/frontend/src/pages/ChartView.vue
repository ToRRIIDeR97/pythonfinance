<script setup>
import { ref, onMounted, watch, computed, nextTick } from 'vue'
import { getIndexHistorical, getSectorHistorical, getSMA, getRSI, getMACD } from '../services/api'
import { useRoute } from 'vue-router'
import { useMarketStore } from '../stores/market'
import TimeframeSelector from '../components/TimeframeSelector.vue'
import MarketSelector from '../components/MarketSelector.vue'
import KeyStatisticsCard from '../components/KeyStatisticsCard.vue'
import IndicatorSelector from '../components/IndicatorSelector.vue'

console.log('=== ChartView component loaded ===')

const route = useRoute()
console.log('Route params:', route.params)
console.log('Route query :', route.query)
const store = useMarketStore()
// Ensure route param is decoded so we don't double-encode (e.g., %5EGSPC -> ^GSPC)
const ticker = computed(() => {
  const t = route.params.ticker || store.selectedTicker
  return typeof t === 'string' ? decodeURIComponent(t) : t
})
const category = computed(() => route.query.category || store.selectedCategory)
const timeframe = computed(() => store.timeframe)
const loading = ref(true)
const error = ref(null)
const hasChartData = ref(false)
const chartOptions = ref({
  rangeSelector: { selected: 1 },
  title: { text: '' },
  series: []
})

const loadData = async () => {
  console.log('loadData() start – ticker:', ticker.value, 'category:', category.value)
  loading.value = true
  error.value = null
  // Guard unsupported categories early to avoid blank charts
  const supported = ['index', 'etf']
  if (!supported.includes(category.value)) {
    loading.value = false
    hasChartData.value = false
    chartOptions.value.series = []
    error.value = `Charting for category '${category.value}' is not supported yet.`
    return
  }
  
  try {
    const hist = category.value === 'etf'
      ? await getSectorHistorical(ticker.value, timeframe.value)
      : await getIndexHistorical(ticker.value, timeframe.value)
    const histArray = Array.isArray(hist) ? hist : (hist?.data || [])
    
    // Fetch selected indicators dynamically
    const indicators = store.indicators || []
    const indicatorFetches = indicators.map(ind => {
      if (ind.type === 'sma') {
        return getSMA(ticker.value, { ...timeframe.value, window: ind.window })
          .then(res => ({ ind, data: res.sma || [] }))
      }
      if (ind.type === 'rsi') {
        return getRSI(ticker.value, { ...timeframe.value, window: ind.window })
          .then(res => ({ ind, data: res.rsi || [] }))
      }
      if (ind.type === 'macd') {
        return getMACD(ticker.value, { ...timeframe.value })
          .then(res => ({ ind, data: res.macd || [] }))
      }
      return Promise.resolve({ ind, data: [] })
    })
    const indicatorResults = await Promise.all(indicatorFetches)

    const ohlc = histArray.map(r => [Date.parse(r.date), r.open, r.high, r.low, r.close])
    
    // Determine which panes we need
    const hasRSI = indicators.some(ind => ind.type === 'rsi')
    const hasMACD = indicators.some(ind => ind.type === 'macd')
    
    const dynamicSeries = []
    indicatorResults.forEach(({ ind, data }) => {
      if (ind.type === 'macd') {
        // MACD has three components: MACD line, signal line, and histogram
        const macdLine = data
          .filter(d => d.macd !== null && d.macd !== undefined)
          .map(d => [Date.parse(d.date), d.macd])
        const signalLine = data
          .filter(d => d.signal !== null && d.signal !== undefined)
          .map(d => [Date.parse(d.date), d.signal])
        const histogram = data
          .filter(d => d.histogram !== null && d.histogram !== undefined)
          .map(d => [Date.parse(d.date), d.histogram])
        
        // Determine MACD yAxis based on other indicators
        const macdYAxis = hasRSI ? 2 : 1
        
        dynamicSeries.push(
          {
            type: 'line',
            name: 'MACD',
            data: macdLine,
            color: '#1f77b4',
            yAxis: macdYAxis,
            tooltip: { valueDecimals: 4 }
          },
          {
            type: 'line',
            name: 'Signal',
            data: signalLine,
            color: '#ff7f0e',
            yAxis: macdYAxis,
            tooltip: { valueDecimals: 4 }
          },
          {
            type: 'column',
            name: 'Histogram',
            data: histogram,
            color: '#2ca02c',
            yAxis: macdYAxis,
            tooltip: { valueDecimals: 4 }
          }
        )
      } else {
        const points = data
          .filter(d => d.value !== null && d.value !== undefined)
          .map(d => [Date.parse(d.date), d.value])
        const base = {
          type: 'line',
          name: `${ind.type.toUpperCase()}(${ind.window})`,
          data: points,
          color: ind.color,
          tooltip: { valueDecimals: 2 }
        }
        if (ind.type === 'rsi') {
          base.yAxis = 1
        }
        dynamicSeries.push(base)
      }
    })

    const volumeSeries = histArray.map(r => [Date.parse(r.date), r.volume])
    console.log('Loaded records:', histArray.length)
    console.log('OHLC points:', ohlc.length, 'Volume points:', volumeSeries.length)
    // Determine if we have data to render
    const indicatorHasData = dynamicSeries.some(s => Array.isArray(s.data) && s.data.length > 0)
    hasChartData.value = (ohlc.length > 0) || indicatorHasData || (volumeSeries.length > 0)
    
    console.log('Chart data check:', {
      ohlcLength: ohlc.length,
      indicatorHasData,
      volumeLength: volumeSeries.length,
      hasChartData: hasChartData.value
    })
    
    // Determine volume yAxis based on other indicators
    let volumeYAxis = 1
    if (hasRSI && hasMACD) {
      volumeYAxis = 3
    } else if (hasRSI || hasMACD) {
      volumeYAxis = 2
    }

    // Build yAxis configuration
    const yAxisConfig = [
      { // 0: Price axis
        labels: { align: 'right', x: -3 },
        title: { text: 'Price' },
        height: '60%',
        lineWidth: 2,
        resize: { enabled: true }
      }
    ]

    if (hasRSI) {
      yAxisConfig.push({
        labels: { align: 'right', x: -3 },
        title: { text: 'RSI' },
        top: '65%',
        height: '15%',
        offset: 0,
        lineWidth: 2,
        min: 0,
        max: 100,
        plotLines: [
          { value: 70, color: 'red', dashStyle: 'shortdash', width: 1 },
          { value: 30, color: 'green', dashStyle: 'shortdash', width: 1 }
        ]
      })
    }

    if (hasMACD) {
      const macdTop = hasRSI ? '82%' : '65%'
      const macdHeight = hasRSI ? '15%' : '30%'
      yAxisConfig.push({
        labels: { align: 'right', x: -3 },
        title: { text: 'MACD' },
        top: macdTop,
        height: macdHeight,
        offset: 0,
        lineWidth: 2
      })
    }

    // Volume axis
    const volumeTop = hasRSI && hasMACD ? '99%' : hasRSI || hasMACD ? '82%' : '65%'
    const volumeHeight = '15%'
    yAxisConfig.push({
      labels: { align: 'right', x: -3 },
      title: { text: 'Volume' },
      top: volumeTop,
      height: volumeHeight,
      offset: 0,
      lineWidth: 2
    })

    chartOptions.value = {
      rangeSelector: { 
        selected: 1,
        enabled: true
      },
      chart: {
        height: 500
      },
      title: { text: `${ticker.value} (${category.value})` },
      series: [
        { 
          type: 'ohlc', 
          name: ticker.value, 
          data: ohlc, 
          tooltip: { valueDecimals: 2 } 
        },
        ...dynamicSeries,
        { 
          type: 'column', 
          name: 'Volume', 
          data: volumeSeries, 
          yAxis: volumeYAxis, 
          color: '#888' 
        }
      ],
      yAxis: yAxisConfig,
      navigator: {
        enabled: true
      },
      scrollbar: {
        enabled: true
      }
    }
    console.log('Chart options series count:', chartOptions.value?.series?.length || 0)
    console.log('Chart options created:', chartOptions.value)
  } catch (e) {
    console.error('Error in loadData:', e)
    error.value = e?.message || 'Failed to load chart data'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  console.log('ChartView mounted!')
  console.log('ChartView onMounted – calling loadData')
  loadData()
})
watch(() => route.fullPath, loadData)
watch(() => timeframe.value, loadData, { deep: true })
watch(() => store.indicators, loadData, { deep: true })
</script>

<template>
  <div class="chart-view">
    <div class="page-header animate-fade-in-down">
      <h2 class="text-3xl font-bold text-primary">{{ ticker }} Chart</h2>
      <p class="text-secondary">Real-time market analysis and technical indicators</p>
    </div>
    
    <div class="toolbar animate-fade-in-up animate-delay-200 stagger-children" style="--stagger-delay: 100ms;">
      <div class="animate-slide-in-left" style="--stagger-index: 0;">
        <TimeframeSelector />
      </div>
      <div class="animate-slide-in-left" style="--stagger-index: 1;">
        <MarketSelector />
      </div>
      <div class="animate-slide-in-left" style="--stagger-index: 2;">
        <IndicatorSelector />
      </div>
    </div>
    
    <div class="animate-fade-in-up animate-delay-400">
      <KeyStatisticsCard />
    </div>
    
    <div v-if="loading" class="loading-state animate-pulse">
      <div class="loading-spinner"></div>
      <p class="text-secondary">Loading chart data...</p>
    </div>

    <div v-else class="chart-container animate-scale-in animate-delay-600">
      <div v-if="error" class="error animate-shake">
        <div class="error-icon">⚠️</div>
        <p>{{ error }}</p>
      </div>
      <div v-else class="chart-wrapper hover-lift">
        <!-- Debug info to help verify reactive state -->
        <div class="text-sm text-secondary" style="margin-bottom: 0.5rem;">
          Series: {{ chartOptions?.series?.length || 0 }} | Title: {{ chartOptions?.title?.text || '' }} | HasData: {{ hasChartData }}
        </div>
        <highcharts 
          v-if="hasChartData && chartOptions.series && chartOptions.series.length > 0" 
          :options="chartOptions"
          :constructor-type="'stockChart'"
        />
        <div v-else class="no-data-message">
          <div class="no-data-icon">📊</div>
          <h3>No Chart Data Available</h3>
          <p>Unable to load chart data for {{ ticker }}. Please try a different symbol or timeframe.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chart-view {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 2rem;
  text-align: center;
}

.toolbar { 
  display: flex; 
  gap: 1rem; 
  align-items: center; 
  margin-bottom: 2rem;
  justify-content: center;
  flex-wrap: wrap;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem;
  text-align: center;
}

.loading-spinner {
  width: 3rem;
  height: 3rem;
  border: 3px solid #e5e7eb;
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.chart-container {
  margin-top: 2rem;
}

.error { 
  color: #ef4444; 
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
}

.error-icon {
  font-size: 1.5rem;
}

.chart-wrapper {
  background: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1rem;
  min-height: 500px;
  transition: all 0.3s ease;
}

.chart-wrapper:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.no-data-message {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--color-text-secondary, #6b7280);
}

.no-data-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.no-data-message h3 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--color-text-primary, #1f2937);
}

.no-data-message p {
  font-size: 1rem;
  opacity: 0.8;
}
</style>
