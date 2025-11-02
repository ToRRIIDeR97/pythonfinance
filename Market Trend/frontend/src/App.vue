<script setup>
import { onMounted } from 'vue'
import { useThemeStore } from './stores/theme'

const theme = useThemeStore()
onMounted(() => {
  // Apply the saved theme on startup
  theme.applyTheme()
})

function toggleTheme() {
  theme.toggleTheme()
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="flex items-center gap-4">
        <h1>📈 Market Trend</h1>
      </div>
      <nav class="flex items-center gap-6">
        <router-link to="/" class="nav-link">Dashboard</router-link>
        <router-link to="/markets" class="nav-link">Markets</router-link>
        <router-link to="/portfolio" class="nav-link">Portfolio</router-link>
        <button class="theme-toggle" @click="toggleTheme">
          {{ theme.theme === 'dark' ? '☀️' : '🌙' }}
        </button>
      </nav>
    </header>
    <main class="content">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.nav-link {
  font-weight: var(--font-medium);
  color: var(--color-text-secondary);
  transition: color var(--transition-fast);
  text-decoration: none;
}

.nav-link:hover {
  color: var(--color-primary);
  text-decoration: none;
}

.nav-link.router-link-active {
  color: var(--color-primary);
  font-weight: var(--font-semibold);
}

.theme-toggle {
  background-color: var(--color-surface-elevated);
  border: 1px solid var(--color-border);
  color: var(--color-text-primary);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  cursor: pointer;
  transition: all var(--transition-fast);
  min-width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.theme-toggle:hover {
  background-color: var(--color-primary-light);
  border-color: var(--color-primary);
  transform: scale(1.05);
}
</style>
