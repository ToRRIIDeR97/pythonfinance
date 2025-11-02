<template>
  <div class="mini-chart-container">
    <div 
      v-if="loading" 
      class="mini-chart-loading"
    >
      <div class="loading-shimmer"></div>
    </div>
    <div 
      v-else-if="error" 
      class="mini-chart-error"
    >
      <span class="error-icon">⚠</span>
    </div>
    <highcharts 
      v-else-if="chartData && chartData.length > 0"
      :options="chartOptions" 
      class="mini-chart"
    />
    <div 
      v-else 
      class="mini-chart-no-data"
    >
      <span class="no-data-icon">📊</span>
    </div>
  </div>
</template>

<script setup>
import { computed, watch, ref, onMounted } from 'vue'
import { getHistoricalData } from '../services/api'

const props = defineProps({
  ticker: {
    type: String,
    required: true
  },
  category: {
    type: String,
    required: true
  },
  period: {
    type: String,
    default: '1mo'
  },
  height: {
    type: Number,
    default: 60
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  lineColor: {
    type: String,
    default: '#3b82f6'
  },
  fillColor: {
    type: String,
    default: 'rgba(59, 130, 246, 0.1)'
  }
})

const chartData = ref([])
const loading = ref(false)
const error = ref(null)

const chartOptions = computed(() => {
  if (!chartData.value || chartData.value.length === 0) return {}
  
  const data = chartData.value.map(point => [point[0], point[4]]) // Use close price
  const isPositive = data.length > 1 && data[data.length - 1][1] >= data[0][1]
  
  return {
    chart: {
      type: 'areaspline',
      height: props.height,
      backgroundColor: 'transparent',
      margin: [2, 2, 2, 2],
      spacing: [0, 0, 0, 0],
      animation: {
        duration: 750,
        easing: 'easeOutQuart'
      }
    },
    title: { text: null },
    legend: { enabled: false },
    credits: { enabled: false },
    xAxis: {
      visible: false,
      type: 'datetime'
    },
    yAxis: {
      visible: false,
      startOnTick: false,
      endOnTick: false
    },
    tooltip: props.showTooltip ? {
      enabled: true,
      outside: true,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      borderColor: 'transparent',
      borderRadius: 6,
      style: {
        color: '#ffffff',
        fontSize: '12px'
      },
      formatter: function() {
        const date = new Date(this.x).toLocaleDateString()
        return `<b>${props.ticker}</b><br/>${date}<br/>$${this.y.toFixed(2)}`
      }
    } : { enabled: false },
    plotOptions: {
      areaspline: {
        lineWidth: 2,
        marker: { enabled: false },
        states: {
          hover: {
            lineWidth: 2,
            marker: { enabled: true, radius: 3 }
          }
        },
        fillOpacity: 0.3,
        animation: {
          duration: 750
        }
      }
    },
    series: [{
      name: props.ticker,
      data: data,
      color: isPositive ? '#10b981' : '#ef4444',
      fillColor: {
        linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
        stops: [
          [0, isPositive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'],
          [1, 'rgba(255, 255, 255, 0)']
        ]
      }
    }]
  }
})

async function fetchChartData() {
  if (!props.ticker || !props.category) return
  
  loading.value = true
  error.value = null
  
  try {
    const resp = await getHistoricalData(
      props.ticker,
      props.category,
      { period: props.period, interval: '1d' }
    )
    const hist = Array.isArray(resp) ? resp : (resp?.data || [])
    if (hist && hist.length > 0) {
      // Convert to [ts, o, h, l, c] tuples for lightweight plotting
      chartData.value = hist.map(r => [Date.parse(r.date), r.open, r.high, r.low, r.close])
    } else {
      chartData.value = []
    }
  } catch (err) {
    console.error(`Failed to fetch mini chart data for ${props.ticker}:`, err)
    error.value = err.message || 'Failed to load chart data'
    chartData.value = []
  } finally {
    loading.value = false
  }
}

// Watch for prop changes
watch([() => props.ticker, () => props.category, () => props.period], fetchChartData, { immediate: false })

onMounted(() => {
  fetchChartData()
})
</script>

<style scoped>
.mini-chart-container {
  width: 100%;
  height: 60px;
  position: relative;
  overflow: hidden;
  border-radius: 6px;
}

.mini-chart {
  width: 100%;
  height: 100%;
}

.mini-chart-loading {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-shimmer {
  width: 80%;
  height: 20px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.mini-chart-error,
.mini-chart-no-data {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-secondary);
  font-size: 18px;
}

.error-icon {
  color: var(--color-danger);
}

.no-data-icon {
  opacity: 0.5;
}

/* Dark mode support */
.dark .loading-shimmer {
  background: linear-gradient(90deg, #374151 25%, #4b5563 50%, #374151 75%);
  background-size: 200% 100%;
}
</style>
