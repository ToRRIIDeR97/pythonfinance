<template>
  <div class="search-filter-container">
    <!-- Search Bar -->
    <div class="search-section">
      <div class="search-input-wrapper">
        <div class="search-input-container">
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Search tickers, companies, or sectors..."
            class="search-input"
            @input="handleSearchInput"
            @focus="showSuggestions = true"
            @blur="hideSuggestions"
          />
          <button
            v-if="searchQuery"
            @click="clearSearch"
            class="clear-search-btn"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <!-- Search Suggestions -->
        <SearchSuggestions
          :search-query="searchQuery"
          :suggestions="suggestions"
          :show-suggestions="showSuggestions"
          :popular-data="data"
          @select="selectSuggestion"
          @quick-action="handleQuickAction"
        />
      </div>

      <!-- Quick Filters -->
      <div class="quick-filters">
        <button
          v-for="filter in quickFilters"
          :key="filter.key"
          @click="toggleQuickFilter(filter.key)"
          :class="['quick-filter-btn', { active: activeQuickFilters.includes(filter.key) }]"
        >
          {{ filter.label }}
        </button>
      </div>
    </div>

    <!-- Advanced Filters Toggle -->
    <div class="filter-controls">
      <button
        @click="showAdvancedFilters = !showAdvancedFilters"
        class="advanced-filter-toggle"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"></polygon>
        </svg>
        Advanced Filters
        <svg 
          :class="['chevron', { rotated: showAdvancedFilters }]"
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor"
        >
          <polyline points="6,9 12,15 18,9"></polyline>
        </svg>
      </button>

      <!-- Saved Filters -->
      <div class="saved-filters">
        <select v-model="selectedSavedFilter" @change="applySavedFilter" class="saved-filter-select">
          <option value="">Saved Filters</option>
          <option v-for="filter in savedFilters" :key="filter.id" :value="filter.id">
            {{ filter.name }}
          </option>
        </select>
        <button @click="showSaveFilterModal = true" class="save-filter-btn" title="Save Current Filter">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
            <polyline points="17,21 17,13 7,13 7,21"></polyline>
            <polyline points="7,3 7,8 15,8"></polyline>
          </svg>
        </button>
      </div>
    </div>

    <!-- Advanced Filters Panel -->
    <div v-if="showAdvancedFilters" class="advanced-filters-panel">
      <div class="filter-grid">
        <!-- Category Filter -->
        <div class="filter-group">
          <label class="filter-label">Category</label>
          <div class="checkbox-group">
            <label v-for="category in categories" :key="category" class="checkbox-label">
              <input
                type="checkbox"
                :value="category"
                v-model="filters.categories"
                @change="applyFilters"
              />
              <span class="checkbox-custom"></span>
              {{ category }}
            </label>
          </div>
        </div>

        <!-- Region Filter -->
        <div class="filter-group">
          <label class="filter-label">Region</label>
          <div class="checkbox-group">
            <label v-for="region in regions" :key="region" class="checkbox-label">
              <input
                type="checkbox"
                :value="region"
                v-model="filters.regions"
                @change="applyFilters"
              />
              <span class="checkbox-custom"></span>
              {{ region }}
            </label>
          </div>
        </div>

        <!-- Price Range Filter -->
        <div class="filter-group">
          <label class="filter-label">Price Range</label>
          <div class="range-inputs">
            <input
              type="number"
              v-model.number="filters.priceRange.min"
              placeholder="Min"
              class="range-input"
              @input="applyFilters"
            />
            <span class="range-separator">to</span>
            <input
              type="number"
              v-model.number="filters.priceRange.max"
              placeholder="Max"
              class="range-input"
              @input="applyFilters"
            />
          </div>
        </div>

        <!-- Performance Filter -->
        <div class="filter-group">
          <label class="filter-label">Performance (%)</label>
          <div class="range-inputs">
            <input
              type="number"
              v-model.number="filters.performanceRange.min"
              placeholder="Min %"
              class="range-input"
              @input="applyFilters"
            />
            <span class="range-separator">to</span>
            <input
              type="number"
              v-model.number="filters.performanceRange.max"
              placeholder="Max %"
              class="range-input"
              @input="applyFilters"
            />
          </div>
        </div>

        <!-- Volume Filter -->
        <div class="filter-group">
          <label class="filter-label">Volume</label>
          <select v-model="filters.volumeRange" @change="applyFilters" class="filter-select">
            <option value="">Any Volume</option>
            <option value="low">Low (< 1M)</option>
            <option value="medium">Medium (1M - 10M)</option>
            <option value="high">High (> 10M)</option>
          </select>
        </div>

        <!-- Sort Options -->
        <div class="filter-group">
          <label class="filter-label">Sort By</label>
          <select v-model="filters.sortBy" @change="applyFilters" class="filter-select">
            <option value="ticker">Ticker</option>
            <option value="name">Name</option>
            <option value="price">Price</option>
            <option value="change">Change</option>
            <option value="changePercent">Change %</option>
            <option value="volume">Volume</option>
          </select>
          <select v-model="filters.sortOrder" @change="applyFilters" class="filter-select">
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </div>
      </div>

      <!-- Filter Actions -->
      <div class="filter-actions">
        <button @click="clearAllFilters" class="clear-filters-btn">
          Clear All Filters
        </button>
        <div class="active-filters-count" v-if="activeFiltersCount > 0">
          {{ activeFiltersCount }} filter{{ activeFiltersCount > 1 ? 's' : '' }} active
        </div>
      </div>
    </div>

    <!-- Save Filter Modal -->
    <div v-if="showSaveFilterModal" class="modal-overlay" @click="showSaveFilterModal = false">
      <div class="modal-content" @click.stop>
        <h3>Save Filter</h3>
        <input
          v-model="newFilterName"
          type="text"
          placeholder="Enter filter name..."
          class="filter-name-input"
          @keyup.enter="saveCurrentFilter"
        />
        <div class="modal-actions">
          <button @click="showSaveFilterModal = false" class="cancel-btn">Cancel</button>
          <button @click="saveCurrentFilter" class="save-btn">Save</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { debounce } from 'lodash-es'
import SearchSuggestions from './SearchSuggestions.vue'

const props = defineProps({
  data: {
    type: Array,
    default: () => []
  },
  placeholder: {
    type: String,
    default: 'Search...'
  }
})

const emit = defineEmits(['filtered', 'search'])

// Search state
const searchQuery = ref('')
const showSuggestions = ref(false)
const suggestions = ref([])

// Filter state
const showAdvancedFilters = ref(false)
const activeQuickFilters = ref([])
const filters = ref({
  categories: [],
  regions: [],
  priceRange: { min: null, max: null },
  performanceRange: { min: null, max: null },
  volumeRange: '',
  sortBy: 'ticker',
  sortOrder: 'asc'
})

// Saved filters
const savedFilters = ref([])
const selectedSavedFilter = ref('')
const showSaveFilterModal = ref(false)
const newFilterName = ref('')

// Quick filters configuration
const quickFilters = ref([
  { key: 'gainers', label: 'Top Gainers' },
  { key: 'losers', label: 'Top Losers' },
  { key: 'active', label: 'Most Active' },
  { key: 'indices', label: 'Indices Only' },
  { key: 'etfs', label: 'ETFs Only' }
])

// Available options
const categories = computed(() => {
  const cats = new Set(props.data.map(item => item.category))
  return Array.from(cats).filter(Boolean)
})

const regions = computed(() => {
  const regs = new Set(props.data.map(item => item.region))
  return Array.from(regs).filter(Boolean)
})

// Active filters count
const activeFiltersCount = computed(() => {
  let count = 0
  if (filters.value.categories.length > 0) count++
  if (filters.value.regions.length > 0) count++
  if (filters.value.priceRange.min !== null || filters.value.priceRange.max !== null) count++
  if (filters.value.performanceRange.min !== null || filters.value.performanceRange.max !== null) count++
  if (filters.value.volumeRange) count++
  if (activeQuickFilters.value.length > 0) count += activeQuickFilters.value.length
  return count
})

// Search functionality
const handleSearchInput = debounce(() => {
  updateSuggestions()
  applyFilters()
}, 300)

const updateSuggestions = () => {
  if (!searchQuery.value.trim()) {
    suggestions.value = []
    return
  }

  const query = searchQuery.value.toLowerCase()
  suggestions.value = props.data
    .filter(item => 
      item.ticker?.toLowerCase().includes(query) ||
      item.name?.toLowerCase().includes(query) ||
      item.category?.toLowerCase().includes(query)
    )
    .slice(0, 8)
}

const selectSuggestion = (suggestion) => {
  if (suggestion.isRecent) {
    searchQuery.value = suggestion.ticker
  } else {
    searchQuery.value = suggestion.ticker
  }
  showSuggestions.value = false
  applyFilters()
}

const handleQuickAction = (action) => {
  toggleQuickFilter(action.key)
}

const hideSuggestions = () => {
  setTimeout(() => {
    showSuggestions.value = false
  }, 200)
}

const clearSearch = () => {
  searchQuery.value = ''
  suggestions.value = []
  applyFilters()
}

// Filter functionality
const toggleQuickFilter = (filterKey) => {
  const index = activeQuickFilters.value.indexOf(filterKey)
  if (index > -1) {
    activeQuickFilters.value.splice(index, 1)
  } else {
    activeQuickFilters.value.push(filterKey)
  }
  applyFilters()
}

const applyFilters = () => {
  let filteredData = [...props.data]

  // Apply search filter
  if (searchQuery.value.trim()) {
    const query = searchQuery.value.toLowerCase()
    filteredData = filteredData.filter(item =>
      item.ticker?.toLowerCase().includes(query) ||
      item.name?.toLowerCase().includes(query) ||
      item.category?.toLowerCase().includes(query)
    )
  }

  // Apply category filter
  if (filters.value.categories.length > 0) {
    filteredData = filteredData.filter(item =>
      filters.value.categories.includes(item.category)
    )
  }

  // Apply region filter
  if (filters.value.regions.length > 0) {
    filteredData = filteredData.filter(item =>
      filters.value.regions.includes(item.region)
    )
  }

  // Apply price range filter
  if (filters.value.priceRange.min !== null || filters.value.priceRange.max !== null) {
    filteredData = filteredData.filter(item => {
      const price = item.price || 0
      const min = filters.value.priceRange.min
      const max = filters.value.priceRange.max
      return (min === null || price >= min) && (max === null || price <= max)
    })
  }

  // Apply performance range filter
  if (filters.value.performanceRange.min !== null || filters.value.performanceRange.max !== null) {
    filteredData = filteredData.filter(item => {
      const perf = item.changePercent || 0
      const min = filters.value.performanceRange.min
      const max = filters.value.performanceRange.max
      return (min === null || perf >= min) && (max === null || perf <= max)
    })
  }

  // Apply volume filter
  if (filters.value.volumeRange) {
    filteredData = filteredData.filter(item => {
      const volume = item.volume || 0
      switch (filters.value.volumeRange) {
        case 'low': return volume < 1000000
        case 'medium': return volume >= 1000000 && volume <= 10000000
        case 'high': return volume > 10000000
        default: return true
      }
    })
  }

  // Apply quick filters
  activeQuickFilters.value.forEach(filterKey => {
    switch (filterKey) {
      case 'gainers':
        filteredData = filteredData.filter(item => (item.changePercent || 0) > 0)
        break
      case 'losers':
        filteredData = filteredData.filter(item => (item.changePercent || 0) < 0)
        break
      case 'active':
        filteredData = filteredData.filter(item => (item.volume || 0) > 1000000)
        break
      case 'indices':
        filteredData = filteredData.filter(item => item.category === 'index')
        break
      case 'etfs':
        filteredData = filteredData.filter(item => item.category === 'etf')
        break
    }
  })

  // Apply sorting
  filteredData.sort((a, b) => {
    const aVal = a[filters.value.sortBy] || 0
    const bVal = b[filters.value.sortBy] || 0
    
    if (typeof aVal === 'string') {
      return filters.value.sortOrder === 'asc' 
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal)
    }
    
    return filters.value.sortOrder === 'asc' 
      ? aVal - bVal 
      : bVal - aVal
  })

  emit('filtered', filteredData)
  emit('search', searchQuery.value)
}

const clearAllFilters = () => {
  searchQuery.value = ''
  activeQuickFilters.value = []
  filters.value = {
    categories: [],
    regions: [],
    priceRange: { min: null, max: null },
    performanceRange: { min: null, max: null },
    volumeRange: '',
    sortBy: 'ticker',
    sortOrder: 'asc'
  }
  applyFilters()
}

// Saved filters functionality
const saveCurrentFilter = () => {
  if (!newFilterName.value.trim()) return

  const filterConfig = {
    id: Date.now().toString(),
    name: newFilterName.value.trim(),
    searchQuery: searchQuery.value,
    activeQuickFilters: [...activeQuickFilters.value],
    filters: JSON.parse(JSON.stringify(filters.value))
  }

  savedFilters.value.push(filterConfig)
  localStorage.setItem('marketDashboardFilters', JSON.stringify(savedFilters.value))
  
  newFilterName.value = ''
  showSaveFilterModal.value = false
}

const applySavedFilter = () => {
  if (!selectedSavedFilter.value) return

  const savedFilter = savedFilters.value.find(f => f.id === selectedSavedFilter.value)
  if (!savedFilter) return

  searchQuery.value = savedFilter.searchQuery || ''
  activeQuickFilters.value = [...(savedFilter.activeQuickFilters || [])]
  filters.value = JSON.parse(JSON.stringify(savedFilter.filters))
  
  applyFilters()
}

// Load saved filters on mount
onMounted(() => {
  const saved = localStorage.getItem('marketDashboardFilters')
  if (saved) {
    try {
      savedFilters.value = JSON.parse(saved)
    } catch (e) {
      console.error('Failed to load saved filters:', e)
    }
  }
})

// Watch for data changes
watch(() => props.data, () => {
  applyFilters()
}, { immediate: true })
</script>

<style scoped>
.search-filter-container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.search-section {
  margin-bottom: 1rem;
}

.search-input-wrapper {
  position: relative;
  margin-bottom: 1rem;
}

.search-input-container {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 1rem;
  width: 20px;
  height: 20px;
  color: #6b7280;
  z-index: 1;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 3rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.2s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.clear-search-btn {
  position: absolute;
  right: 0.75rem;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0.25rem;
  color: #6b7280;
  transition: color 0.2s ease;
}

.clear-search-btn:hover {
  color: #374151;
}

.clear-search-btn svg {
  width: 16px;
  height: 16px;
}



.quick-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.quick-filter-btn {
  padding: 0.5rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 20px;
  background: white;
  color: #6b7280;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.quick-filter-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.quick-filter-btn.active {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.filter-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.advanced-filter-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease;
}

.advanced-filter-toggle:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.advanced-filter-toggle svg {
  width: 16px;
  height: 16px;
}

.chevron {
  transition: transform 0.2s ease;
}

.chevron.rotated {
  transform: rotate(180deg);
}

.saved-filters {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.saved-filter-select {
  padding: 0.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  color: #374151;
}

.save-filter-btn {
  padding: 0.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.save-filter-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.save-filter-btn svg {
  width: 16px;
  height: 16px;
}

.advanced-filters-panel {
  border-top: 1px solid #e5e7eb;
  padding-top: 1.5rem;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.filter-label {
  font-weight: 600;
  color: #374151;
  font-size: 0.875rem;
}

.checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  color: #6b7280;
}

.checkbox-label input[type="checkbox"] {
  display: none;
}

.checkbox-custom {
  width: 16px;
  height: 16px;
  border: 2px solid #e5e7eb;
  border-radius: 3px;
  position: relative;
  transition: all 0.2s ease;
}

.checkbox-label input[type="checkbox"]:checked + .checkbox-custom {
  background: #3b82f6;
  border-color: #3b82f6;
}

.checkbox-label input[type="checkbox"]:checked + .checkbox-custom::after {
  content: '';
  position: absolute;
  top: 1px;
  left: 4px;
  width: 4px;
  height: 8px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.range-inputs {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.range-input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  font-size: 0.875rem;
}

.range-separator {
  color: #6b7280;
  font-size: 0.875rem;
}

.filter-select {
  padding: 0.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  color: #374151;
  font-size: 0.875rem;
}

.filter-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.clear-filters-btn {
  padding: 0.5rem 1rem;
  border: 1px solid #ef4444;
  border-radius: 6px;
  background: white;
  color: #ef4444;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-filters-btn:hover {
  background: #ef4444;
  color: white;
}

.active-filters-count {
  font-size: 0.875rem;
  color: #6b7280;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.modal-content {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  width: 90%;
  max-width: 400px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

.modal-content h3 {
  margin: 0 0 1rem 0;
  color: #1f2937;
}

.filter-name-input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  margin-bottom: 1.5rem;
  font-size: 1rem;
}

.modal-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.cancel-btn, .save-btn {
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn {
  border: 1px solid #e5e7eb;
  background: white;
  color: #6b7280;
}

.cancel-btn:hover {
  background: #f9fafb;
}

.save-btn {
  border: 1px solid #3b82f6;
  background: #3b82f6;
  color: white;
}

.save-btn:hover {
  background: #2563eb;
}

@media (max-width: 768px) {
  .filter-grid {
    grid-template-columns: 1fr;
  }
  
  .filter-controls {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }
  
  .saved-filters {
    justify-content: center;
  }
}
</style>