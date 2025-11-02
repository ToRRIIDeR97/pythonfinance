import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from './pages/Dashboard.vue'
import Markets from './pages/Markets.vue'
import Portfolio from './pages/Portfolio.vue'
import ChartView from './pages/ChartView.vue'

const routes = [
  { path: '/', name: 'dashboard', component: Dashboard },
  { path: '/markets', name: 'markets', component: Markets },
  { path: '/portfolio', name: 'portfolio', component: Portfolio },
  { path: '/chart/:ticker', name: 'chart', component: ChartView, props: route => ({ ticker: route.params.ticker, category: route.query.category || 'index' }) },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router

