<template>
  <div class="markets">
    <!-- Page Header -->
    <div class="page-header mb-8 animate-fade-in-down">
      <h1 class="text-4xl font-bold text-primary mb-2">Markets Overview</h1>
      <p class="text-lg text-secondary">Explore and analyze global financial markets</p>
    </div>

    <!-- Controls Section -->
    <div class="controls-section mb-8 animate-fade-in-up animate-delay-200">
      <div class="controls-grid stagger-children" style="--stagger-delay: 100ms;">
        <!-- Market Filter -->
        <div class="control-group animate-slide-in-left" style="--stagger-index: 0;">
          <label class="control-label">Market Type</label>
          <select v-model="selectedMarketType" class="form-select focus-ring">
            <option value="all">All Markets</option>
            <option value="indices">Indices</option>
            <option value="etfs">ETFs</option>
            <option value="stocks">Stocks</option>
            <option value="crypto">Cryptocurrency</option>
          </select>
        </div>

        <!-- Search Bar -->
        <div class="control-group animate-slide-in-left" style="--stagger-index: 1;">
          <label class="control-label">Search Symbol</label>
          <div class="search-input-wrapper">
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search by symbol or name..."
              class="form-input search-input focus-ring"
            />
            <div class="search-icon">🔍</div>
          </div>
        </div>

        <!-- Sort Options -->
        <div class="control-group animate-slide-in-left" style="--stagger-index: 2;">
          <label class="control-label">Sort By</label>
          <select v-model="sortBy" class="form-select focus-ring">
            <option value="symbol">Symbol</option>
            <option value="name">Name</option>
            <option value="price">Price</option>
            <option value="change">Change</option>
            <option value="changePercent">% Change</option>
            <option value="volume">Volume</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p class="text-secondary">Loading market data...</p>
    </div>

    <!-- Error State -->
    <div v-if="error" class="error-state">
      <div class="error-icon">⚠️</div>
      <h3 class="text-xl font-semibold text-primary mb-2">Unable to Load Market Data</h3>
      <p class="text-secondary">{{ error }}</p>
      <button @click="fetchMarketData" class="btn btn-primary mt-4">Retry</button>
    </div>

    <!-- Market Data Content -->
    <div v-if="!loading && !error" class="market-content animate-fade-in-up animate-delay-400">
      <!-- Market Data Table -->
      <div class="market-table-section mb-8">
        <div class="table-header mb-4 animate-slide-in-left">
          <h2 class="text-2xl font-semibold text-primary">Market Data</h2>
          <div class="table-actions">
            <button class="btn btn-outline btn-sm hover-lift">Export CSV</button>
            <button class="btn btn-outline btn-sm hover-lift">Refresh Data</button>
          </div>
        </div>

        <DataTable
          :data="filteredAndSortedData"
          :columns="tableColumns"
          :loading="loading"
          :error="error"
          searchable
          sortable
          paginated
          :page-size="20"
          search-placeholder="Search stocks..."
          empty-title="No stocks found"
          empty-description="Try adjusting your search or filter criteria."
          @row-click="handleRowClick"
          @retry="fetchMarketData"
        >
          <template #cell-symbol="{ value }">
            <span class="symbol-text">{{ value }}</span>
          </template>
          
          <template #cell-change="{ value }">
            <span :class="getChangeClass(value)">
              {{ formatChange(value) }}
            </span>
          </template>
          
          <template #cell-changePercent="{ value }">
            <span :class="getChangeClass(value)">
              {{ formatPercent(value) }}
            </span>
          </template>
          
          <template #cell-volume="{ value }">
            {{ formatVolume(value) }}
          </template>
          
          <template #cell-actions="{ item }">
            <div class="action-buttons">
              <button @click.stop="viewChart(item)" class="btn btn-primary btn-xs">Chart</button>
              <button @click.stop="addToWatchlist(item)" class="btn btn-outline btn-xs">Watch</button>
            </div>
          </template>
        </DataTable>
      </div>

      <!-- Market Heatmap Section -->
      <div class="heatmap-section">
        <div class="section-header mb-6">
          <h2 class="text-2xl font-semibold text-primary">Market Heatmap</h2>
          <p class="text-secondary">Visual representation of market performance</p>
        </div>
        <div class="heatmap-container">
          <div class="heatmap-placeholder">
            <div class="heatmap-icon">📊</div>
            <h3 class="text-lg font-medium text-primary mb-2">Interactive Heatmap</h3>
            <p class="text-secondary">Coming soon - Real-time sector performance visualization</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import DataTable from '../components/DataTable.vue'

export default {
  name: 'Markets',
  components: {
    DataTable
  },
  data() {
    return {
      loading: false,
      error: null,
      selectedMarketType: 'all',
      searchQuery: '',
      sortBy: 'symbol',
      sortDirection: 'asc',
      currentPage: 1,
      itemsPerPage: 20,
      marketData: [
        // Sample data - in real app this would come from API
        { symbol: 'AAPL', name: 'Apple Inc.', price: 150.25, change: 2.50, changePercent: 1.69, volume: 50000000, type: 'stocks' },
        { symbol: 'MSFT', name: 'Microsoft Corporation', price: 300.80, change: -1.20, changePercent: -0.40, volume: 30000000, type: 'stocks' },
        { symbol: '^GSPC', name: 'S&P 500', price: 4200.50, change: 15.30, changePercent: 0.37, volume: 0, type: 'indices' },
        { symbol: 'XLK', name: 'Technology Select Sector SPDR Fund', price: 145.60, change: 1.80, changePercent: 1.25, volume: 8000000, type: 'etfs' },
        { symbol: 'BTC-USD', name: 'Bitcoin USD', price: 45000.00, change: -500.00, changePercent: -1.10, volume: 25000000000, type: 'crypto' },
      ],
      tableColumns: [
        { key: 'symbol', title: 'Symbol', sortable: true, width: '100px' },
        { key: 'name', title: 'Name', sortable: true, width: '250px' },
        { key: 'price', title: 'Price', sortable: true, align: 'right', type: 'currency', width: '120px' },
        { key: 'change', title: 'Change', sortable: true, align: 'right', width: '100px' },
        { key: 'changePercent', title: '% Change', sortable: true, align: 'right', width: '100px' },
        { key: 'volume', title: 'Volume', sortable: true, align: 'right', type: 'number', width: '120px' },
        { key: 'actions', title: 'Actions', sortable: false, align: 'center', width: '150px' }
      ]
    }
  },
  computed: {
    filteredAndSortedData() {
      let filtered = this.marketData;

      // Filter by market type
      if (this.selectedMarketType !== 'all') {
        filtered = filtered.filter(item => item.type === this.selectedMarketType);
      }

      // Filter by search query
      if (this.searchQuery) {
        const query = this.searchQuery.toLowerCase();
        filtered = filtered.filter(item => 
          item.symbol.toLowerCase().includes(query) || 
          item.name.toLowerCase().includes(query)
        );
      }

      // Sort data
      filtered.sort((a, b) => {
        let aVal = a[this.sortBy];
        let bVal = b[this.sortBy];
        
        if (typeof aVal === 'string') {
          aVal = aVal.toLowerCase();
          bVal = bVal.toLowerCase();
        }
        
        if (this.sortDirection === 'asc') {
          return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        } else {
          return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
        }
      });

      return filtered;
    },
    totalItems() {
      return this.filteredAndSortedData.length;
    },
    totalPages() {
      return Math.ceil(this.totalItems / this.itemsPerPage);
    },
    startIndex() {
      return (this.currentPage - 1) * this.itemsPerPage;
    },
    endIndex() {
      return Math.min(this.startIndex + this.itemsPerPage, this.totalItems);
    },
    paginatedData() {
      return this.filteredAndSortedData.slice(this.startIndex, this.endIndex);
    }
  },
  methods: {
    fetchMarketData() {
      this.loading = true;
      this.error = null;
      
      // Simulate API call
      setTimeout(() => {
        this.loading = false;
      }, 1000);
    },
    setSortBy(field) {
      if (this.sortBy === field) {
        this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.sortBy = field;
        this.sortDirection = 'asc';
      }
    },
    getSortClass(field) {
      if (this.sortBy !== field) return '';
      return this.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc';
    },
    getChangeClass(value) {
      return value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral';
    },
    formatPrice(price) {
      return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    },
    formatChange(change) {
      const sign = change > 0 ? '+' : '';
      return `${sign}${change.toFixed(2)}`;
    },
    formatPercent(percent) {
      const sign = percent > 0 ? '+' : '';
      return `${sign}${percent.toFixed(2)}%`;
    },
    formatVolume(volume) {
      if (volume === 0) return 'N/A';
      if (volume >= 1000000000) return `${(volume / 1000000000).toFixed(1)}B`;
      if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
      if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
      return volume.toLocaleString();
    },
    viewChart(item) {
      // Pass the item type as category to ChartView so it knows which API to call
      const typeToCategory = {
        indices: 'index',
        etfs: 'etf',
        stocks: 'stock',
        crypto: 'crypto'
      };
      const category = typeToCategory[item.type] || 'index';
      this.$router.push({
        path: `/chart/${encodeURIComponent(item.symbol)}`,
        query: { category }
      });
    },
    addToWatchlist(item) {
      // TODO: Implement watchlist functionality
      console.log('Adding to watchlist:', item.symbol);
    },
    previousPage() {
      if (this.currentPage > 1) {
        this.currentPage--;
      }
    },
    nextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++;
      }
    }
  },
  mounted() {
    this.fetchMarketData();
  }
}
</script>

<style scoped>
.markets {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  padding: var(--space-8) 0;
}

.controls-section {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
}

.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--space-6);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.control-label {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
  font-size: var(--text-sm);
}

.search-input-wrapper {
  position: relative;
}

.search-input {
  padding-right: var(--space-10);
}

.search-icon {
  position: absolute;
  right: var(--space-3);
  top: 50%;
  transform: translateY(-50%);
  color: var(--color-text-secondary);
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-16) 0;
  gap: var(--space-4);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--color-border);
  border-top: 3px solid var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.error-state {
  text-align: center;
  padding: var(--space-16) 0;
}

.error-icon {
  font-size: 3rem;
  margin-bottom: var(--space-4);
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.table-actions {
  display: flex;
  gap: var(--space-2);
}

.table-container {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.market-table {
  width: 100%;
  border-collapse: collapse;
}

.market-table th {
  background-color: var(--gray-50);
  padding: var(--space-4) var(--space-6);
  text-align: left;
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

.dark .market-table th {
  background-color: var(--color-surface-elevated);
}

.market-table th.sortable {
  cursor: pointer;
  user-select: none;
  transition: background-color var(--transition-fast);
}

.market-table th.sortable:hover {
  background-color: var(--gray-100);
}

.dark .market-table th.sortable:hover {
  background-color: var(--gray-800);
}

.sort-indicator {
  margin-left: var(--space-1);
  opacity: 0.5;
  font-size: var(--text-xs);
}

.sort-indicator.sort-asc {
  opacity: 1;
}

.sort-indicator.sort-desc {
  opacity: 1;
  transform: rotate(180deg);
}

.market-table td {
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--color-border);
  vertical-align: middle;
}

.table-row:hover {
  background-color: var(--gray-50);
}

.dark .table-row:hover {
  background-color: var(--color-surface-elevated);
}

.symbol-text {
  font-family: var(--font-family-mono);
  font-weight: var(--font-bold);
  color: var(--color-primary);
}

.positive {
  color: var(--color-success);
}

.negative {
  color: var(--color-danger);
}

.neutral {
  color: var(--color-text-secondary);
}

.action-buttons {
  display: flex;
  gap: var(--space-2);
  justify-content: center;
}

.pagination-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-controls {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.page-info {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
}

.heatmap-section {
  margin-top: var(--space-12);
}

.section-header {
  text-align: center;
}

.heatmap-container {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-16);
  box-shadow: var(--shadow-sm);
}

.heatmap-placeholder {
  text-align: center;
  color: var(--color-text-secondary);
}

.heatmap-icon {
  font-size: 4rem;
  margin-bottom: var(--space-4);
}

/* Responsive Design */
@media (max-width: 1024px) {
  .controls-grid {
    grid-template-columns: 1fr;
  }
  
  .table-actions {
    flex-direction: column;
  }
  
  .pagination-section {
    flex-direction: column;
    gap: var(--space-4);
  }
}

@media (max-width: 768px) {
  .table-container {
    overflow-x: auto;
  }
  
  .market-table {
    min-width: 800px;
  }
  
  .action-buttons {
    flex-direction: column;
  }
}
</style>
