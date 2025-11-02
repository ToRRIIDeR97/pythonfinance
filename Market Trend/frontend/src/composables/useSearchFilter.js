import { ref, computed, watch } from 'vue'
import { debounce } from 'lodash-es'

export function useSearchFilter(initialData = []) {
  // Search state
  const searchQuery = ref('')
  const searchResults = ref([])
  
  // Filter state
  const filters = ref({
    categories: [],
    regions: [],
    priceRange: { min: null, max: null },
    performanceRange: { min: null, max: null },
    volumeRange: '',
    sortBy: 'ticker',
    sortOrder: 'asc'
  })
  
  const quickFilters = ref([])
  const savedFilters = ref([])
  
  // Data state
  const originalData = ref(initialData)
  const filteredData = ref([])
  
  // Computed properties
  const hasActiveFilters = computed(() => {
    return searchQuery.value.trim() !== '' ||
           filters.value.categories.length > 0 ||
           filters.value.regions.length > 0 ||
           filters.value.priceRange.min !== null ||
           filters.value.priceRange.max !== null ||
           filters.value.performanceRange.min !== null ||
           filters.value.performanceRange.max !== null ||
           filters.value.volumeRange !== '' ||
           quickFilters.value.length > 0
  })
  
  const activeFiltersCount = computed(() => {
    let count = 0
    if (searchQuery.value.trim()) count++
    if (filters.value.categories.length > 0) count++
    if (filters.value.regions.length > 0) count++
    if (filters.value.priceRange.min !== null || filters.value.priceRange.max !== null) count++
    if (filters.value.performanceRange.min !== null || filters.value.performanceRange.max !== null) count++
    if (filters.value.volumeRange) count++
    count += quickFilters.value.length
    return count
  })
  
  const availableCategories = computed(() => {
    const categories = new Set(originalData.value.map(item => item.category))
    return Array.from(categories).filter(Boolean)
  })
  
  const availableRegions = computed(() => {
    const regions = new Set(originalData.value.map(item => item.region))
    return Array.from(regions).filter(Boolean)
  })
  
  // Search functionality
  const performSearch = (query) => {
    if (!query.trim()) {
      searchResults.value = []
      return originalData.value
    }
    
    const searchTerm = query.toLowerCase()
    const results = originalData.value.filter(item =>
      item.ticker?.toLowerCase().includes(searchTerm) ||
      item.name?.toLowerCase().includes(searchTerm) ||
      item.category?.toLowerCase().includes(searchTerm) ||
      item.region?.toLowerCase().includes(searchTerm)
    )
    
    searchResults.value = results
    return results
  }
  
  const debouncedSearch = debounce((query) => {
    searchQuery.value = query
    applyAllFilters()
  }, 300)
  
  // Filter functionality
  const applyFilters = (data) => {
    let result = [...data]
    
    // Apply category filter
    if (filters.value.categories.length > 0) {
      result = result.filter(item =>
        filters.value.categories.includes(item.category)
      )
    }
    
    // Apply region filter
    if (filters.value.regions.length > 0) {
      result = result.filter(item =>
        filters.value.regions.includes(item.region)
      )
    }
    
    // Apply price range filter
    if (filters.value.priceRange.min !== null || filters.value.priceRange.max !== null) {
      result = result.filter(item => {
        const price = parseFloat(item.price) || 0
        const min = filters.value.priceRange.min
        const max = filters.value.priceRange.max
        return (min === null || price >= min) && (max === null || price <= max)
      })
    }
    
    // Apply performance range filter
    if (filters.value.performanceRange.min !== null || filters.value.performanceRange.max !== null) {
      result = result.filter(item => {
        const perf = parseFloat(item.changePercent) || 0
        const min = filters.value.performanceRange.min
        const max = filters.value.performanceRange.max
        return (min === null || perf >= min) && (max === null || perf <= max)
      })
    }
    
    // Apply volume filter
    if (filters.value.volumeRange) {
      result = result.filter(item => {
        const volume = parseInt(item.volume) || 0
        switch (filters.value.volumeRange) {
          case 'low': return volume < 1000000
          case 'medium': return volume >= 1000000 && volume <= 10000000
          case 'high': return volume > 10000000
          default: return true
        }
      })
    }
    
    return result
  }
  
  const applyQuickFilters = (data) => {
    let result = [...data]
    
    quickFilters.value.forEach(filterKey => {
      switch (filterKey) {
        case 'gainers':
          result = result.filter(item => (parseFloat(item.changePercent) || 0) > 0)
          break
        case 'losers':
          result = result.filter(item => (parseFloat(item.changePercent) || 0) < 0)
          break
        case 'active':
          result = result.filter(item => (parseInt(item.volume) || 0) > 1000000)
          break
        case 'indices':
          result = result.filter(item => item.category === 'index')
          break
        case 'etfs':
          result = result.filter(item => item.category === 'etf')
          break
      }
    })
    
    return result
  }
  
  const applySorting = (data) => {
    return [...data].sort((a, b) => {
      const aVal = a[filters.value.sortBy] || 0
      const bVal = b[filters.value.sortBy] || 0
      
      if (typeof aVal === 'string') {
        return filters.value.sortOrder === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      }
      
      const aNum = parseFloat(aVal) || 0
      const bNum = parseFloat(bVal) || 0
      
      return filters.value.sortOrder === 'asc'
        ? aNum - bNum
        : bNum - aNum
    })
  }
  
  const applyAllFilters = () => {
    let result = performSearch(searchQuery.value)
    result = applyFilters(result)
    result = applyQuickFilters(result)
    result = applySorting(result)
    
    filteredData.value = result
    return result
  }
  
  // Filter management
  const addQuickFilter = (filterKey) => {
    if (!quickFilters.value.includes(filterKey)) {
      quickFilters.value.push(filterKey)
      applyAllFilters()
    }
  }
  
  const removeQuickFilter = (filterKey) => {
    const index = quickFilters.value.indexOf(filterKey)
    if (index > -1) {
      quickFilters.value.splice(index, 1)
      applyAllFilters()
    }
  }
  
  const toggleQuickFilter = (filterKey) => {
    if (quickFilters.value.includes(filterKey)) {
      removeQuickFilter(filterKey)
    } else {
      addQuickFilter(filterKey)
    }
  }
  
  const clearAllFilters = () => {
    searchQuery.value = ''
    quickFilters.value = []
    filters.value = {
      categories: [],
      regions: [],
      priceRange: { min: null, max: null },
      performanceRange: { min: null, max: null },
      volumeRange: '',
      sortBy: 'ticker',
      sortOrder: 'asc'
    }
    applyAllFilters()
  }
  
  const updateFilter = (filterType, value) => {
    if (filterType in filters.value) {
      filters.value[filterType] = value
      applyAllFilters()
    }
  }
  
  // Saved filters functionality
  const saveFilter = (name) => {
    const filterConfig = {
      id: Date.now().toString(),
      name: name.trim(),
      searchQuery: searchQuery.value,
      quickFilters: [...quickFilters.value],
      filters: JSON.parse(JSON.stringify(filters.value)),
      createdAt: new Date().toISOString()
    }
    
    savedFilters.value.push(filterConfig)
    localStorage.setItem('marketDashboardFilters', JSON.stringify(savedFilters.value))
    
    return filterConfig
  }
  
  const loadFilter = (filterId) => {
    const savedFilter = savedFilters.value.find(f => f.id === filterId)
    if (!savedFilter) return false
    
    searchQuery.value = savedFilter.searchQuery || ''
    quickFilters.value = [...(savedFilter.quickFilters || [])]
    filters.value = JSON.parse(JSON.stringify(savedFilter.filters))
    
    applyAllFilters()
    return true
  }
  
  const deleteFilter = (filterId) => {
    const index = savedFilters.value.findIndex(f => f.id === filterId)
    if (index > -1) {
      savedFilters.value.splice(index, 1)
      localStorage.setItem('marketDashboardFilters', JSON.stringify(savedFilters.value))
      return true
    }
    return false
  }
  
  const loadSavedFilters = () => {
    try {
      const saved = localStorage.getItem('marketDashboardFilters')
      if (saved) {
        savedFilters.value = JSON.parse(saved)
      }
    } catch (error) {
      console.error('Failed to load saved filters:', error)
      savedFilters.value = []
    }
  }
  
  // Data management
  const updateData = (newData) => {
    originalData.value = newData
    applyAllFilters()
  }
  
  const addData = (newItems) => {
    originalData.value = [...originalData.value, ...newItems]
    applyAllFilters()
  }
  
  const removeData = (predicate) => {
    originalData.value = originalData.value.filter(item => !predicate(item))
    applyAllFilters()
  }
  
  // Suggestions for search
  const getSearchSuggestions = (query, limit = 8) => {
    if (!query.trim()) return []
    
    const searchTerm = query.toLowerCase()
    return originalData.value
      .filter(item =>
        item.ticker?.toLowerCase().includes(searchTerm) ||
        item.name?.toLowerCase().includes(searchTerm) ||
        item.category?.toLowerCase().includes(searchTerm)
      )
      .slice(0, limit)
      .map(item => ({
        ...item,
        matchType: item.ticker?.toLowerCase().includes(searchTerm) ? 'ticker' :
                  item.name?.toLowerCase().includes(searchTerm) ? 'name' : 'category'
      }))
  }
  
  // Initialize
  const initialize = (data = []) => {
    originalData.value = data
    loadSavedFilters()
    applyAllFilters()
  }
  
  // Watch for data changes
  watch(() => originalData.value, () => {
    applyAllFilters()
  }, { deep: true })
  
  return {
    // State
    searchQuery,
    searchResults,
    filters,
    quickFilters,
    savedFilters,
    originalData,
    filteredData,
    
    // Computed
    hasActiveFilters,
    activeFiltersCount,
    availableCategories,
    availableRegions,
    
    // Methods
    debouncedSearch,
    performSearch,
    applyAllFilters,
    addQuickFilter,
    removeQuickFilter,
    toggleQuickFilter,
    clearAllFilters,
    updateFilter,
    saveFilter,
    loadFilter,
    deleteFilter,
    loadSavedFilters,
    updateData,
    addData,
    removeData,
    getSearchSuggestions,
    initialize
  }
}

export default useSearchFilter