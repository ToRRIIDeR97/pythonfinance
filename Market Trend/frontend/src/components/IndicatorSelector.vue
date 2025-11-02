<template>
  <div class="indicator-selector">
    <div class="group">
      <span class="label">Overlay</span>
      <label>
        <input type="checkbox" :checked="isSelected('sma', 14)" @change="() => toggle('sma', 14, '#ff7f0e')" />
        SMA 14
      </label>
      <label>
        <input type="checkbox" :checked="isSelected('sma', 50)" @change="() => toggle('sma', 50, '#d62728')" />
        SMA 50
      </label>
      <label>
        <input type="checkbox" :checked="isSelected('sma', 200)" @change="() => toggle('sma', 200, '#2ca02c')" />
        SMA 200
      </label>
    </div>
    <div class="group">
      <span class="label">Pane</span>
      <label>
        <input type="checkbox" :checked="isSelected('rsi', 14)" @change="() => toggle('rsi', 14, '#1f77b4')" />
        RSI 14
      </label>
      <label>
        <input type="checkbox" :checked="isSelected('macd', 12)" @change="() => toggle('macd', 12, '#9467bd')" />
        MACD
      </label>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useMarketStore } from '../stores/market'

const market = useMarketStore()
const selected = computed(() => market.indicators)

function isSelected(type, window) {
  return selected.value.some((ind) => ind.type === type && ind.window === window)
}

function toggle(type, window, color) {
  if (isSelected(type, window)) {
    market.removeIndicator({ type, window })
  } else {
    market.addIndicator({ type, window, color })
  }
}
</script>

<style scoped>
.indicator-selector {
  display: flex;
  gap: 1rem;
  align-items: center;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--card-bg);
}
.group {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}
.label {
  font-weight: 600;
  color: var(--text-color);
}
label {
  color: var(--text-color);
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}
input[type='checkbox'] {
  accent-color: var(--primary-color);
}
</style>

