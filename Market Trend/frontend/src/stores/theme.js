import { defineStore } from 'pinia'

export const useThemeStore = defineStore('theme', {
  state: () => ({
    theme: (typeof localStorage !== 'undefined' && localStorage.getItem('theme')) || 'light',
  }),
  actions: {
    applyTheme() {
      const isDark = this.theme === 'dark'
      if (typeof document !== 'undefined') {
        document.documentElement.classList.toggle('dark', isDark)
      }
    },
    setTheme(t) {
      this.theme = t
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('theme', t)
      }
      this.applyTheme()
    },
    toggleTheme() {
      this.setTheme(this.theme === 'dark' ? 'light' : 'dark')
    }
  }
})
