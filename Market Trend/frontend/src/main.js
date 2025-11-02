import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'
import HighchartsVue from 'highcharts-vue'
import Highcharts from 'highcharts'
import 'highcharts/modules/stock'
import 'highcharts/modules/treemap'
import 'highcharts/modules/heatmap'
import { createPinia } from 'pinia'

const app = createApp(App)
const pinia = createPinia()
app.use(router)
app.use(HighchartsVue)
app.use(pinia)
app.mount('#app')
