<template>
  <div class="data-table-container">
    <!-- Table Controls -->
    <div v-if="showControls" class="table-controls">
      <div class="table-search" v-if="searchable">
        <input
          v-model="searchQuery"
          type="text"
          :placeholder="searchPlaceholder"
          class="search-input"
        />
        <span class="search-icon">🔍</span>
      </div>
      
      <div class="table-actions">
        <slot name="actions"></slot>
      </div>
    </div>

    <!-- Loading State -->
    <LoadingSpinner 
      v-if="loading" 
      :message="loadingMessage"
      size="medium"
    />

    <!-- Error State -->
    <ErrorMessage 
      v-else-if="error" 
      :message="error"
      type="error"
      :actions="[{ label: 'Retry', handler: () => $emit('retry') }]"
    />

    <!-- Empty State -->
    <div v-else-if="filteredData.length === 0" class="empty-state">
      <div class="empty-icon">📊</div>
      <h3 class="empty-title">{{ emptyTitle }}</h3>
      <p class="empty-description">{{ emptyDescription }}</p>
      <slot name="empty-actions"></slot>
    </div>

    <!-- Table -->
    <div v-else class="table-wrapper" :class="{ 'responsive': responsive }">
      <table class="data-table" :class="tableClass">
        <thead>
          <tr>
            <th 
              v-for="column in columns" 
              :key="column.key"
              :class="getHeaderClass(column)"
              @click="handleSort(column)"
              :style="getColumnStyle(column)"
            >
              <div class="header-content">
                <span>{{ column.title }}</span>
                <span v-if="column.sortable" class="sort-indicator">
                  <span 
                    class="sort-arrow"
                    :class="{
                      'active': sortBy === column.key && sortOrder === 'asc',
                      'asc': true
                    }"
                  >▲</span>
                  <span 
                    class="sort-arrow"
                    :class="{
                      'active': sortBy === column.key && sortOrder === 'desc',
                      'desc': true
                    }"
                  >▼</span>
                </span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr 
            v-for="(item, index) in paginatedData" 
            :key="getRowKey(item, index)"
            :class="getRowClass(item, index)"
            @click="handleRowClick(item, index)"
          >
            <td 
              v-for="column in columns" 
              :key="column.key"
              :class="getCellClass(column, item)"
              :style="getColumnStyle(column)"
            >
              <slot 
                :name="`cell-${column.key}`" 
                :item="item" 
                :row="item"
                :value="getCellValue(item, column.key)"
                :index="index"
                v-if="item"
              >
                <span v-if="column.type === 'currency'" class="currency-value">
                  {{ formatCurrency(getCellValue(item, column.key)) }}
                </span>
                <span v-else-if="column.type === 'percentage'" class="percentage-value">
                  {{ formatPercentage(getCellValue(item, column.key)) }}
                </span>
                <span v-else-if="column.type === 'date'" class="date-value">
                  {{ formatDate(getCellValue(item, column.key)) }}
                </span>
                <span v-else-if="column.type === 'number'" class="number-value">
                  {{ formatNumber(getCellValue(item, column.key)) }}
                </span>
                <span v-else>
                  {{ getCellValue(item, column.key) }}
                </span>
              </slot>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div v-if="paginated && filteredData.length > 0" class="table-pagination">
      <div class="pagination-info">
        Showing {{ startIndex + 1 }}-{{ endIndex }} of {{ filteredData.length }} items
      </div>
      <div class="pagination-controls">
        <button 
          @click="goToPage(currentPage - 1)"
          :disabled="currentPage === 1"
          class="btn btn-secondary btn-sm"
        >
          Previous
        </button>
        
        <div class="page-numbers">
          <button
            v-for="page in visiblePages"
            :key="page"
            @click="goToPage(page)"
            :class="['btn', 'btn-sm', page === currentPage ? 'btn-primary' : 'btn-secondary']"
          >
            {{ page }}
          </button>
        </div>
        
        <button 
          @click="goToPage(currentPage + 1)"
          :disabled="currentPage === totalPages"
          class="btn btn-secondary btn-sm"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import LoadingSpinner from './LoadingSpinner.vue';
import ErrorMessage from './ErrorMessage.vue';

export default {
  name: 'DataTable',
  components: {
    LoadingSpinner,
    ErrorMessage
  },
  props: {
    data: {
      type: Array,
      default: () => []
    },
    columns: {
      type: Array,
      required: true
    },
    loading: {
      type: Boolean,
      default: false
    },
    error: {
      type: String,
      default: ''
    },
    loadingMessage: {
      type: String,
      default: 'Loading data...'
    },
    emptyTitle: {
      type: String,
      default: 'No data available'
    },
    emptyDescription: {
      type: String,
      default: 'There are no items to display at this time.'
    },
    searchable: {
      type: Boolean,
      default: true
    },
    searchPlaceholder: {
      type: String,
      default: 'Search...'
    },
    sortable: {
      type: Boolean,
      default: true
    },
    paginated: {
      type: Boolean,
      default: true
    },
    pageSize: {
      type: Number,
      default: 10
    },
    responsive: {
      type: Boolean,
      default: true
    },
    striped: {
      type: Boolean,
      default: true
    },
    hoverable: {
      type: Boolean,
      default: true
    },
    clickableRows: {
      type: Boolean,
      default: false
    },
    showControls: {
      type: Boolean,
      default: true
    },
    rowKey: {
      type: String,
      default: 'id'
    }
  },
  emits: ['row-click', 'sort', 'search', 'retry'],
  data() {
    return {
      searchQuery: '',
      sortBy: '',
      sortOrder: 'asc',
      currentPage: 1
    };
  },
  computed: {
    tableClass() {
      return {
        'striped': this.striped,
        'hoverable': this.hoverable,
        'clickable': this.clickableRows
      };
    },
    filteredData() {
      let filtered = [...this.data];
      
      // Apply search filter
      if (this.searchQuery.trim()) {
        const query = this.searchQuery.toLowerCase();
        filtered = filtered.filter(item => {
          return this.columns.some(column => {
            const value = this.getCellValue(item, column.key);
            return String(value).toLowerCase().includes(query);
          });
        });
      }
      
      // Apply sorting
      if (this.sortBy) {
        filtered.sort((a, b) => {
          const aVal = this.getCellValue(a, this.sortBy);
          const bVal = this.getCellValue(b, this.sortBy);
          
          let comparison = 0;
          if (aVal > bVal) comparison = 1;
          if (aVal < bVal) comparison = -1;
          
          return this.sortOrder === 'desc' ? -comparison : comparison;
        });
      }
      
      return filtered;
    },
    totalPages() {
      return Math.ceil(this.filteredData.length / this.pageSize);
    },
    paginatedData() {
      if (!this.paginated) return this.filteredData;
      
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      return this.filteredData.slice(start, end);
    },
    startIndex() {
      return (this.currentPage - 1) * this.pageSize;
    },
    endIndex() {
      return Math.min(this.startIndex + this.pageSize, this.filteredData.length);
    },
    visiblePages() {
      const pages = [];
      const maxVisible = 5;
      let start = Math.max(1, this.currentPage - Math.floor(maxVisible / 2));
      let end = Math.min(this.totalPages, start + maxVisible - 1);
      
      if (end - start + 1 < maxVisible) {
        start = Math.max(1, end - maxVisible + 1);
      }
      
      for (let i = start; i <= end; i++) {
        pages.push(i);
      }
      
      return pages;
    }
  },
  watch: {
    searchQuery() {
      this.currentPage = 1;
      this.$emit('search', this.searchQuery);
    },
    data() {
      this.currentPage = 1;
    }
  },
  methods: {
    getCellValue(item, key) {
      return key.split('.').reduce((obj, k) => obj?.[k], item) ?? '';
    },
    getRowKey(item, index) {
      return this.getCellValue(item, this.rowKey) || index;
    },
    getHeaderClass(column) {
      return {
        'sortable': column.sortable !== false && this.sortable,
        'sorted': this.sortBy === column.key,
        [`align-${column.align || 'left'}`]: true
      };
    },
    getRowClass(item, index) {
      return {
        'clickable': this.clickableRows
      };
    },
    getCellClass(column, item) {
      return {
        [`align-${column.align || 'left'}`]: true,
        [`type-${column.type || 'text'}`]: true
      };
    },
    getColumnStyle(column) {
      const style = {};
      if (column.width) style.width = column.width;
      if (column.minWidth) style.minWidth = column.minWidth;
      return style;
    },
    handleSort(column) {
      if (column.sortable === false || !this.sortable) return;
      
      if (this.sortBy === column.key) {
        this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
      } else {
        this.sortBy = column.key;
        this.sortOrder = 'asc';
      }
      
      this.$emit('sort', { column: column.key, order: this.sortOrder });
    },
    handleRowClick(item, index) {
      if (this.clickableRows) {
        this.$emit('row-click', { item, index });
      }
    },
    goToPage(page) {
      if (page >= 1 && page <= this.totalPages) {
        this.currentPage = page;
      }
    },
    formatCurrency(value) {
      if (value == null || isNaN(value)) return '-';
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
      }).format(value);
    },
    formatPercentage(value) {
      if (value == null || isNaN(value)) return '-';
      return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }).format(value / 100);
    },
    formatNumber(value) {
      if (value == null || isNaN(value)) return '-';
      return new Intl.NumberFormat('en-US').format(value);
    },
    formatDate(value) {
      if (!value) return '-';
      return new Date(value).toLocaleDateString();
    }
  }
}
</script>

<style scoped>
.data-table-container {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
  overflow: hidden;
}

.table-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-4);
  border-bottom: 1px solid var(--color-border);
  gap: var(--space-4);
}

.table-search {
  position: relative;
  flex: 1;
  max-width: 300px;
}

.search-input {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  padding-right: var(--space-10);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
}

.search-icon {
  position: absolute;
  right: var(--space-3);
  top: 50%;
  transform: translateY(-50%);
  color: var(--color-text-secondary);
}

.table-actions {
  display: flex;
  gap: var(--space-2);
}

.table-wrapper {
  overflow-x: auto;
}

.table-wrapper.responsive {
  -webkit-overflow-scrolling: touch;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: var(--space-3) var(--space-4);
  text-align: left;
  border-bottom: 1px solid var(--color-border);
}

.data-table th {
  background-color: var(--gray-50);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  position: sticky;
  top: 0;
  z-index: 1;
}

.dark .data-table th {
  background-color: var(--gray-800);
}

.data-table th.sortable {
  cursor: pointer;
  user-select: none;
  transition: background-color var(--transition-fast);
}

.data-table th.sortable:hover {
  background-color: var(--gray-100);
}

.dark .data-table th.sortable:hover {
  background-color: var(--gray-700);
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
}

.sort-indicator {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.sort-arrow {
  font-size: 8px;
  color: var(--color-text-secondary);
  transition: color var(--transition-fast);
}

.sort-arrow.active {
  color: var(--color-primary);
}

.data-table.striped tbody tr:nth-child(even) {
  background-color: var(--gray-25);
}

.dark .data-table.striped tbody tr:nth-child(even) {
  background-color: var(--gray-850);
}

.data-table.hoverable tbody tr:hover {
  background-color: var(--gray-50);
}

.dark .data-table.hoverable tbody tr:hover {
  background-color: var(--gray-800);
}

.data-table.clickable tbody tr.clickable {
  cursor: pointer;
}

/* Alignment classes */
.align-left { text-align: left; }
.align-center { text-align: center; }
.align-right { text-align: right; }

/* Type-specific styling */
.type-currency,
.type-number,
.type-percentage {
  font-family: var(--font-family-mono);
  font-weight: var(--font-medium);
}

.currency-value {
  color: var(--color-success);
}

.percentage-value.positive {
  color: var(--color-success);
}

.percentage-value.negative {
  color: var(--color-danger);
}

.empty-state {
  text-align: center;
  padding: var(--space-12);
  color: var(--color-text-secondary);
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: var(--space-4);
}

.empty-title {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  margin-bottom: var(--space-2);
}

.empty-description {
  font-size: var(--text-base);
  margin-bottom: var(--space-6);
}

.table-pagination {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-4);
  border-top: 1px solid var(--color-border);
  background-color: var(--gray-25);
}

.dark .table-pagination {
  background-color: var(--gray-850);
}

.pagination-info {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
}

.pagination-controls {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.page-numbers {
  display: flex;
  gap: var(--space-1);
}

/* Responsive design */
@media (max-width: 768px) {
  .table-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .table-search {
    max-width: none;
  }
  
  .table-pagination {
    flex-direction: column;
    gap: var(--space-3);
  }
  
  .pagination-controls {
    justify-content: center;
  }
  
  .data-table th,
  .data-table td {
    padding: var(--space-2) var(--space-3);
    font-size: var(--text-sm);
  }
}

/* Scrollbar styling */
.table-wrapper::-webkit-scrollbar {
  height: 6px;
}

.table-wrapper::-webkit-scrollbar-track {
  background: var(--gray-100);
}

.table-wrapper::-webkit-scrollbar-thumb {
  background: var(--gray-300);
  border-radius: var(--radius-full);
}

.table-wrapper::-webkit-scrollbar-thumb:hover {
  background: var(--gray-400);
}

.dark .table-wrapper::-webkit-scrollbar-track {
  background: var(--gray-700);
}

.dark .table-wrapper::-webkit-scrollbar-thumb {
  background: var(--gray-600);
}

.dark .table-wrapper::-webkit-scrollbar-thumb:hover {
  background: var(--gray-500);
}
</style>