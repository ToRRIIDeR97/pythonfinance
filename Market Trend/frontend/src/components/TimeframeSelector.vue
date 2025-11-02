<script setup>
import { useMarketStore } from '../stores/market'
const store = useMarketStore()

const options = [
  { label: '5D', period: '5d', interval: '1d' },
  { label: '1M', period: '1mo', interval: '1d' },
  { label: '3M', period: '3mo', interval: '1d' },
  { label: '6M', period: '6mo', interval: '1d' },
  { label: '1Y', period: '1y', interval: '1d' },
  { label: '2Y', period: '2y', interval: '1d' },
]

function select(opt) {
  store.setTimeframe({ period: opt.period, interval: opt.interval })
}
</script>

<template>
  <div class="tf">
    <span>Timeframe:</span>
    <div class="buttons">
      <button v-for="opt in options" :key="opt.label" @click="select(opt)">{{ opt.label }}</button>
    </div>
  </div>
  
</template>

<style scoped>
.tf { display: flex; align-items: center; gap: 8px; }
.buttons { display: flex; gap: 6px; }
button { padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 6px; background: white; cursor: pointer; }
button:hover { background: #f5f5f5; }
</style>

