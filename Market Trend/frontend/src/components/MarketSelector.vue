<script setup>
import { useMarketStore } from '../stores/market'
import { ref, onMounted } from 'vue'
import { listIndices, listSectorETFs } from '../services/api'

const store = useMarketStore()
const indices = ref([])
const etfs = ref([])

onMounted(async () => {
  const [idx, sec] = await Promise.all([listIndices(), listSectorETFs()])
  indices.value = idx
  etfs.value = sec
})

function selectTicker(category, ticker) {
  store.setSelection({ category, ticker })
}
</script>

<template>
  <div class="sel">
    <div>
      <h4>Indices</h4>
      <div class="list">
        <button v-for="i in indices" :key="i.ticker" @click="selectTicker('index', i.ticker)">{{ i.ticker }}</button>
      </div>
    </div>
    <div>
      <h4>Sector ETFs</h4>
      <div class="list">
        <button v-for="e in etfs" :key="e.ticker" @click="selectTicker('etf', e.ticker)">{{ e.ticker }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sel { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.list { display: flex; flex-wrap: wrap; gap: 6px; }
button { padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 6px; background: white; cursor: pointer; }
button:hover { background: #f5f5f5; }
</style>

