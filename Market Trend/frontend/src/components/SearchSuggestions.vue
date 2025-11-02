<template>
  <div v-if="showSuggestions" class="suggestions-container">
    <!-- Search Results -->
    <div v-if="searchSuggestions.length > 0" class="suggestions-section">
      <div class="suggestions-header">
        <svg class="suggestions-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.35-4.35"></path>
        </svg>
        <span>Search Results</span>
      </div>
      <div class="suggestions-list">
        <div
          v-for="suggestion in searchSuggestions"
          :key="suggestion.ticker"
          @click="selectSuggestion(suggestion)"
          class="suggestion-item"
        >
          <div class="suggestion-main">
            <div class="suggestion-ticker">
              <span v-html="highlightMatch(suggestion.ticker, searchQuery)"></span>
            </div>
            <div class="suggestion-name">
              <span v-html="highlightMatch(suggestion.name, searchQuery)"></span>
            </div>
          </div>
          <div class="suggestion-meta">
            <span class="suggestion-category">{{ suggestion.category }}</span>
            <span v-if="suggestion.region" class="suggestion-region">{{ suggestion.region }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Searches -->
    <div v-if="recentSearches.length > 0 && !searchQuery" class="suggestions-section">
      <div class="suggestions-header">
        <svg class="suggestions-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"></path>
        </svg>
        <span>Recent Searches</span>
        <button @click="clearRecentSearches" class="clear-recent-btn">
          Clear
        </button>
      </div>
      <div class="suggestions-list">
        <div
          v-for="recent in recentSearches"
          :key="recent.query"
          @click="selectRecentSearch(recent.query)"
          class="suggestion-item recent-item"
        >
          <div class="suggestion-main">
            <div class="suggestion-ticker">{{ recent.query }}</div>
            <div class="suggestion-time">{{ formatTime(recent.timestamp) }}</div>
          </div>
          <button
            @click.stop="removeRecentSearch(recent.query)"
            class="remove-recent-btn"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Popular Tickers -->
    <div v-if="popularTickers.length > 0 && !searchQuery" class="suggestions-section">
      <div class="suggestions-header">
        <svg class="suggestions-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"></polygon>
        </svg>
        <span>Popular Tickers</span>
      </div>
      <div class="suggestions-list">
        <div
          v-for="ticker in popularTickers"
          :key="ticker.ticker"
          @click="selectSuggestion(ticker)"
          class="suggestion-item popular-item"
        >
          <div class="suggestion-main">
            <div class="suggestion-ticker">{{ ticker.ticker }}</div>
            <div class="suggestion-name">{{ ticker.name }}</div>
          </div>
          <div class="suggestion-meta">
            <span class="suggestion-category">{{ ticker.category }}</span>
            <div class="suggestion-performance" :class="getPerformanceClass(ticker.changePercent)">
              {{ formatPercentage(ticker.changePercent) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div v-if="!searchQuery" class="suggestions-section">
      <div class="suggestions-header">
        <svg class="suggestions-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"></polygon>
        </svg>
        <span>Quick Filters</span>
      </div>
      <div class="quick-actions">
        <button
          v-for="action in quickActions"
          :key="action.key"
          @click="selectQuickAction(action)"
          class="quick-action-btn"
        >
          <svg v-html="action.icon" class="action-icon"></svg>
          {{ action.label }}
        </button>
      </div>
    </div>

    <!-- No Results -->
    <div v-if="searchQuery && searchSuggestions.length === 0" class="no-results-section">
      <div class="no-results-content">
        <svg class="no-results-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.35-4.35"></path>
        </svg>
        <p class="no-results-text">No results found for "{{ searchQuery }}"</p>
        <p class="no-results-suggestion">Try searching for ticker symbols, company names, or sectors</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const props = defineProps({
  searchQuery: {
    type: String,
    default: ''
  },
  suggestions: {
    type: Array,
    default: () => []
  },
  showSuggestions: {
    type: Boolean,
    default: false
  },
  popularData: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['select', 'quickAction'])

// Recent searches state
const recentSearches = ref([])
const maxRecentSearches = 5

// Popular tickers (top performers, most active, etc.)
const popularTickers = computed(() => {
  return props.popularData
    .filter(item => item.volume && item.changePercent !== undefined)
    .sort((a, b) => (b.volume || 0) - (a.volume || 0))
    .slice(0, 6)
})

// Search suggestions with highlighting
const searchSuggestions = computed(() => {
  return props.suggestions.slice(0, 8)
})

// Quick actions
const quickActions = ref([
  {
    key: 'gainers',
    label: 'Top Gainers',
    icon: '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
  },
  {
    key: 'losers',
    label: 'Top Losers',
    icon: '<polyline points="1 18 10.5 8.5 15.5 13.5 23 6"></polyline>'
  },
  {
    key: 'active',
    label: 'Most Active',
    icon: '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>'
  },
  {
    key: 'indices',
    label: 'Indices Only',
    icon: '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><rect x="7" y="7" width="10" height="10" rx="1" ry="1"></rect>'
  }
])

// Methods
const highlightMatch = (text, query) => {
  if (!query || !text) return text
  
  const regex = new RegExp(`(${query})`, 'gi')
  return text.replace(regex, '<mark>$1</mark>')
}

const selectSuggestion = (suggestion) => {
  addToRecentSearches(suggestion.ticker)
  emit('select', suggestion)
}

const selectRecentSearch = (query) => {
  emit('select', { ticker: query, isRecent: true })
}

const selectQuickAction = (action) => {
  emit('quickAction', action)
}

const addToRecentSearches = (query) => {
  if (!query.trim()) return
  
  // Remove if already exists
  const existingIndex = recentSearches.value.findIndex(item => item.query === query)
  if (existingIndex > -1) {
    recentSearches.value.splice(existingIndex, 1)
  }
  
  // Add to beginning
  recentSearches.value.unshift({
    query,
    timestamp: Date.now()
  })
  
  // Limit to max recent searches
  if (recentSearches.value.length > maxRecentSearches) {
    recentSearches.value = recentSearches.value.slice(0, maxRecentSearches)
  }
  
  // Save to localStorage
  localStorage.setItem('marketSearchRecent', JSON.stringify(recentSearches.value))
}

const removeRecentSearch = (query) => {
  const index = recentSearches.value.findIndex(item => item.query === query)
  if (index > -1) {
    recentSearches.value.splice(index, 1)
    localStorage.setItem('marketSearchRecent', JSON.stringify(recentSearches.value))
  }
}

const clearRecentSearches = () => {
  recentSearches.value = []
  localStorage.removeItem('marketSearchRecent')
}

const loadRecentSearches = () => {
  try {
    const saved = localStorage.getItem('marketSearchRecent')
    if (saved) {
      recentSearches.value = JSON.parse(saved)
    }
  } catch (error) {
    console.error('Failed to load recent searches:', error)
    recentSearches.value = []
  }
}

const formatTime = (timestamp) => {
  const now = Date.now()
  const diff = now - timestamp
  
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return `${Math.floor(diff / 86400000)}d ago`
}

const formatPercentage = (value) => {
  if (value === undefined || value === null) return ''
  const num = parseFloat(value)
  return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`
}

const getPerformanceClass = (value) => {
  if (value === undefined || value === null) return ''
  const num = parseFloat(value)
  return num >= 0 ? 'positive' : 'negative'
}

// Watch for search query changes to add to recent searches
watch(() => props.searchQuery, (newQuery) => {
  if (newQuery && newQuery.length > 2) {
    // Debounce adding to recent searches
    setTimeout(() => {
      if (props.searchQuery === newQuery) {
        addToRecentSearches(newQuery)
      }
    }, 1000)
  }
})

onMounted(() => {
  loadRecentSearches()
})
</script>

<style scoped>
.suggestions-container {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  max-height: 500px;
  overflow-y: auto;
}

.suggestions-section {
  border-bottom: 1px solid #f3f4f6;
}

.suggestions-section:last-child {
  border-bottom: none;
}

.suggestions-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: #f9fafb;
  border-bottom: 1px solid #f3f4f6;
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
}

.suggestions-icon {
  width: 16px;
  height: 16px;
  color: #6b7280;
}

.clear-recent-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.clear-recent-btn:hover {
  background: #e5e7eb;
  color: #374151;
}

.suggestions-list {
  max-height: 200px;
  overflow-y: auto;
}

.suggestion-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  cursor: pointer;
  border-bottom: 1px solid #f9fafb;
  transition: background-color 0.2s ease;
}

.suggestion-item:hover {
  background-color: #f9fafb;
}

.suggestion-item:last-child {
  border-bottom: none;
}

.suggestion-main {
  flex: 1;
  min-width: 0;
}

.suggestion-ticker {
  font-weight: 600;
  color: #1f2937;
  font-size: 0.875rem;
}

.suggestion-name {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.suggestion-time {
  font-size: 0.75rem;
  color: #9ca3af;
  margin-top: 0.25rem;
}

.suggestion-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.suggestion-category {
  font-size: 0.75rem;
  color: #6b7280;
  text-transform: uppercase;
  background: #f3f4f6;
  padding: 0.125rem 0.375rem;
  border-radius: 4px;
}

.suggestion-region {
  font-size: 0.75rem;
  color: #6b7280;
}

.suggestion-performance {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.125rem 0.375rem;
  border-radius: 4px;
}

.suggestion-performance.positive {
  color: #059669;
  background: #d1fae5;
}

.suggestion-performance.negative {
  color: #dc2626;
  background: #fee2e2;
}

.remove-recent-btn {
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.remove-recent-btn:hover {
  background: #f3f4f6;
  color: #6b7280;
}

.remove-recent-btn svg {
  width: 14px;
  height: 14px;
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.5rem;
  padding: 1rem;
}

.quick-action-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 0.875rem;
}

.quick-action-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
  background: #f8faff;
}

.action-icon {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.no-results-section {
  padding: 2rem 1rem;
}

.no-results-content {
  text-align: center;
}

.no-results-icon {
  width: 48px;
  height: 48px;
  color: #9ca3af;
  margin: 0 auto 1rem;
}

.no-results-text {
  font-size: 1rem;
  color: #374151;
  margin: 0 0 0.5rem 0;
}

.no-results-suggestion {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
}

/* Highlight styling */
:deep(mark) {
  background: #fef3c7;
  color: #92400e;
  padding: 0.125rem 0.25rem;
  border-radius: 3px;
  font-weight: 600;
}

/* Scrollbar styling */
.suggestions-container::-webkit-scrollbar,
.suggestions-list::-webkit-scrollbar {
  width: 6px;
}

.suggestions-container::-webkit-scrollbar-track,
.suggestions-list::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.suggestions-container::-webkit-scrollbar-thumb,
.suggestions-list::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

.suggestions-container::-webkit-scrollbar-thumb:hover,
.suggestions-list::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

@media (max-width: 768px) {
  .quick-actions {
    grid-template-columns: 1fr;
  }
  
  .suggestion-meta {
    flex-direction: column;
    align-items: flex-end;
    gap: 0.25rem;
  }
}
</style>