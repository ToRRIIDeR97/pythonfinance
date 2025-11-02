<template>
  <div class="portfolio">
    <!-- Page Header -->
    <div class="page-header mb-8 animate-fade-in-down">
      <div class="header-content">
        <h1 class="text-4xl font-bold text-primary mb-2">Portfolio</h1>
        <p class="text-lg text-secondary">Track your investments and performance</p>
      </div>
      <div class="header-actions">
        <button @click="showAddPosition = true" class="btn btn-primary hover-lift">
          Add Position
        </button>
      </div>
    </div>

    <!-- Loading State -->
    <LoadingSpinner 
      v-if="loading" 
      :centered="true" 
      message="Loading portfolio data..." 
    />

    <!-- Error State -->
    <ErrorMessage 
      v-if="error" 
      type="error"
      title="Unable to Load Portfolio"
      :message="error"
      :actions="[{ label: 'Retry', action: fetchPortfolioData, variant: 'primary' }]"
    />

    <!-- Portfolio Content -->
    <div v-if="!loading && !error" class="portfolio-content animate-fade-in-up animate-delay-200">
      <!-- Portfolio Summary Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 stagger-children" style="--stagger-delay: 150ms;">
        <Card variant="default" size="medium" :hoverable="true" class="animate-fade-in-up hover-lift" style="--stagger-index: 0;">
          <template #body>
            <div class="flex items-center space-x-4">
              <div class="text-3xl">💰</div>
              <div>
                <h3 class="text-sm font-medium text-secondary mb-1">Total Value</h3>
                <p class="text-2xl font-bold text-primary">${{ formatCurrency(portfolioSummary.totalValue) }}</p>
                <p class="text-xs text-secondary">Portfolio Balance</p>
              </div>
            </div>
          </template>
        </Card>

        <Card variant="default" size="medium" :hoverable="true" class="animate-fade-in-up hover-lift" style="--stagger-index: 1;">
          <template #body>
            <div class="flex items-center space-x-4">
              <div class="text-3xl">📈</div>
              <div>
                <h3 class="text-sm font-medium text-secondary mb-1">Today's Change</h3>
                <p class="text-2xl font-bold" :class="getChangeClass(portfolioSummary.todayChange)">
                  {{ formatChange(portfolioSummary.todayChange) }}
                </p>
                <p class="text-xs" :class="getChangeClass(portfolioSummary.todayChangePercent)">
                  {{ formatPercent(portfolioSummary.todayChangePercent) }}
                </p>
              </div>
            </div>
          </template>
        </Card>

        <Card variant="default" size="medium" :hoverable="true" class="animate-fade-in-up hover-lift" style="--stagger-index: 2;">
          <template #body>
            <div class="flex items-center space-x-4">
              <div class="text-3xl">🎯</div>
              <div>
                <h3 class="text-sm font-medium text-secondary mb-1">Total Return</h3>
                <p class="text-2xl font-bold" :class="getChangeClass(portfolioSummary.totalReturn)">
                  {{ formatChange(portfolioSummary.totalReturn) }}
                </p>
                <p class="text-xs" :class="getChangeClass(portfolioSummary.totalReturnPercent)">
                  {{ formatPercent(portfolioSummary.totalReturnPercent) }}
                </p>
              </div>
            </div>
          </template>
        </Card>
      </div>

      <!-- Portfolio Chart -->
      <div class="chart-section mb-8 animate-fade-in-up animate-delay-400">
        <div class="section-header mb-6 animate-slide-in-left">
          <h2 class="text-2xl font-semibold text-primary">Portfolio Performance</h2>
          <div class="chart-controls stagger-children" style="--stagger-delay: 50ms;">
            <button 
              v-for="(period, index) in chartPeriods" 
              :key="period.value"
              @click="selectedPeriod = period.value"
              :class="['btn', 'btn-sm', 'hover-lift', 'animate-fade-in-up', selectedPeriod === period.value ? 'btn-primary' : 'btn-outline']"
              :style="`--stagger-index: ${index}`"
            >
              {{ period.label }}
            </button>
          </div>
        </div>
        <div class="chart-container animate-scale-in animate-delay-500">
          <div class="chart-placeholder">
            <div class="chart-icon">📊</div>
            <h3 class="text-lg font-medium text-primary mb-2">Portfolio Chart</h3>
            <p class="text-secondary">Interactive chart showing portfolio performance over {{ selectedPeriod }}</p>
          </div>
        </div>
      </div>

      <!-- Holdings Table -->
      <div class="holdings-section mb-8 animate-fade-in-up animate-delay-600">
        <div class="section-header mb-6 animate-slide-in-left">
          <h2 class="text-2xl font-semibold text-primary">Holdings</h2>
          <button @click="showAddPosition = true" class="btn btn-primary hover-lift">
            + Add Position
          </button>
        </div>

        <DataTable
          :data="holdings"
          :columns="holdingsColumns"
          :loading="loading"
          :error="error"
          :searchable="true"
          :sortable="true"
          :paginated="true"
          :items-per-page="10"
          search-placeholder="Search holdings..."
        >
          <template #cell-symbol="{ row }">
            <div class="flex flex-col" v-if="row">
              <span class="font-semibold text-primary">{{ row.symbol || 'N/A' }}</span>
              <span class="text-xs text-secondary">{{ row.name || 'N/A' }}</span>
            </div>
          </template>
          
          <template #cell-pnl="{ row }">
            <span v-if="row" :class="getChangeClass(row.pnl)">
              {{ formatChange(row.pnl) }}
            </span>
          </template>
          
          <template #cell-pnlPercent="{ row }">
            <span v-if="row" :class="getChangeClass(row.pnlPercent)">
              {{ formatPercent(row.pnlPercent) }}
            </span>
          </template>
          
          <template #cell-actions="{ row }">
            <div v-if="row" class="flex space-x-2">
              <button @click="viewChart(row)" class="btn btn-primary btn-xs">Chart</button>
              <button @click="editPosition(row)" class="btn btn-outline btn-xs">Edit</button>
              <button @click="sellPosition(row)" class="btn btn-danger btn-xs">Sell</button>
            </div>
          </template>
        </DataTable>
      </div>

      <!-- Bottom Section: Asset Allocation & Recent Activity -->
      <div class="bottom-section">
        <div class="bottom-grid">
          <!-- Asset Allocation -->
          <div class="allocation-section">
            <div class="section-header mb-6">
              <h2 class="text-xl font-semibold text-primary">Asset Allocation</h2>
            </div>
            <div class="allocation-container">
              <div class="allocation-chart">
                <div class="chart-placeholder-small">
                  <div class="pie-chart-icon">🥧</div>
                  <p class="text-sm text-secondary">Pie Chart</p>
                </div>
              </div>
              <div class="allocation-legend">
                <div v-for="allocation in assetAllocation" :key="allocation.sector" class="legend-item">
                  <div class="legend-color" :style="{ backgroundColor: allocation.color }"></div>
                  <span class="legend-label">{{ allocation.sector }}</span>
                  <span class="legend-value">{{ allocation.percentage }}%</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Recent Activity -->
          <div class="activity-section">
            <div class="section-header mb-6">
              <h2 class="text-xl font-semibold text-primary">Recent Activity</h2>
            </div>
            <div class="activity-list">
              <div v-for="activity in recentActivity" :key="activity.id" class="activity-item">
                <div class="activity-icon" :class="activity.type">
                  {{ getActivityIcon(activity.type) }}
                </div>
                <div class="activity-content">
                  <p class="activity-description">{{ activity.description }}</p>
                  <p class="activity-date">{{ formatDate(activity.date) }}</p>
                </div>
                <div class="activity-amount" :class="getChangeClass(activity.amount)">
                  {{ formatChange(activity.amount) }}
                </div>
              </div>
            </div>
            <div class="activity-footer">
              <button class="btn btn-outline btn-sm">View All Transactions</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Add Position Modal -->
    <Modal
      v-if="showAddPosition"
      title="Add New Position"
      size="medium"
      @close="showAddPosition = false"
    >
      <template #body>
        <p class="text-secondary mb-4">Position management functionality coming soon...</p>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-primary mb-2">Symbol</label>
            <input type="text" class="input" placeholder="Enter stock symbol (e.g., AAPL)" />
          </div>
          <div>
            <label class="block text-sm font-medium text-primary mb-2">Shares</label>
            <input type="number" class="input" placeholder="Number of shares" />
          </div>
          <div>
            <label class="block text-sm font-medium text-primary mb-2">Purchase Price</label>
            <input type="number" class="input" placeholder="Price per share" step="0.01" />
          </div>
        </div>
      </template>
      <template #footer>
        <div class="flex justify-end space-x-3">
          <button @click="showAddPosition = false" class="btn btn-outline">Cancel</button>
          <button class="btn btn-primary">Add Position</button>
        </div>
      </template>
    </Modal>
  </div>
</template>

<script>
import Card from '../components/Card.vue'
import DataTable from '../components/DataTable.vue'
import Modal from '../components/Modal.vue'
import LoadingSpinner from '../components/LoadingSpinner.vue'
import ErrorMessage from '../components/ErrorMessage.vue'

export default {
  name: 'Portfolio',
  components: {
    Card,
    DataTable,
    Modal,
    LoadingSpinner,
    ErrorMessage
  },
  data() {
    return {
      loading: false,
      error: null,
      showAddPosition: false,
      selectedPeriod: '1Y',
      chartPeriods: [
        { label: '1D', value: '1D' },
        { label: '1W', value: '1W' },
        { label: '1M', value: '1M' },
        { label: '3M', value: '3M' },
        { label: '1Y', value: '1Y' },
        { label: 'All', value: 'All' }
      ],
      portfolioSummary: {
        totalValue: 125430.50,
        todayChange: 1234.56,
        todayChangePercent: 0.99,
        totalReturn: 25430.50,
        totalReturnPercent: 25.43
      },
      holdings: [
        {
          symbol: 'AAPL',
          name: 'Apple Inc.',
          shares: 100,
          avgCost: 140.00,
          currentPrice: 150.25,
          marketValue: 15025.00,
          pnl: 1025.00,
          pnlPercent: 7.32
        },
        {
          symbol: 'MSFT',
          name: 'Microsoft Corporation',
          shares: 50,
          avgCost: 280.00,
          currentPrice: 300.80,
          marketValue: 15040.00,
          pnl: 1040.00,
          pnlPercent: 7.43
        },
        {
          symbol: 'GOOGL',
          name: 'Alphabet Inc.',
          shares: 25,
          avgCost: 2400.00,
          currentPrice: 2650.00,
          marketValue: 66250.00,
          pnl: 6250.00,
          pnlPercent: 10.42
        },
        {
          symbol: 'TSLA',
          name: 'Tesla Inc.',
          shares: 30,
          avgCost: 800.00,
          currentPrice: 750.00,
          marketValue: 22500.00,
          pnl: -1500.00,
          pnlPercent: -6.25
        }
      ],
      holdingsColumns: [
        { key: 'symbol', title: 'Symbol', sortable: true, width: '100px' },
        { key: 'name', title: 'Name', sortable: true, width: '200px' },
        { key: 'shares', title: 'Shares', sortable: true, align: 'right', type: 'number', width: '100px' },
        { key: 'avgCost', title: 'Avg Cost', sortable: true, align: 'right', type: 'currency', width: '120px' },
        { key: 'currentPrice', title: 'Current Price', sortable: true, align: 'right', type: 'currency', width: '120px' },
        { key: 'marketValue', title: 'Market Value', sortable: true, align: 'right', type: 'currency', width: '140px' },
        { key: 'pnl', title: 'P&L', sortable: true, align: 'right', width: '120px' },
        { key: 'pnlPercent', title: 'P&L %', sortable: true, align: 'right', width: '100px' },
        { key: 'actions', title: 'Actions', sortable: false, align: 'center', width: '120px' }
      ],
      assetAllocation: [
        { sector: 'Technology', percentage: 45, color: '#3B82F6' },
        { sector: 'Finance', percentage: 25, color: '#10B981' },
        { sector: 'Healthcare', percentage: 20, color: '#F59E0B' },
        { sector: 'Other', percentage: 10, color: '#6B7280' }
      ],
      recentActivity: [
        {
          id: 1,
          type: 'buy',
          description: 'Bought 10 AAPL @ $148.50',
          date: new Date('2024-01-15'),
          amount: 1485.00
        },
        {
          id: 2,
          type: 'sell',
          description: 'Sold 25 TSLA @ $220.00',
          date: new Date('2024-01-14'),
          amount: 5500.00
        },
        {
          id: 3,
          type: 'dividend',
          description: 'Dividend from MSFT',
          date: new Date('2024-01-13'),
          amount: 50.00
        },
        {
          id: 4,
          type: 'buy',
          description: 'Bought 5 GOOGL @ $2,600.00',
          date: new Date('2024-01-12'),
          amount: 13000.00
        }
      ]
    }
  },
  methods: {
    fetchPortfolioData() {
      this.loading = true;
      this.error = null;
      
      // Simulate API call
      setTimeout(() => {
        this.loading = false;
      }, 1000);
    },
    getChangeClass(value) {
      return value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral';
    },
    formatCurrency(value) {
      return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    },
    formatPrice(price) {
      return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    },
    formatChange(change) {
      const sign = change > 0 ? '+' : '';
      return `${sign}$${Math.abs(change).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    },
    formatPercent(percent) {
      const sign = percent > 0 ? '+' : '';
      return `${sign}${percent.toFixed(2)}%`;
    },
    formatDate(date) {
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    },
    getActivityIcon(type) {
      const icons = {
        buy: '📈',
        sell: '📉',
        dividend: '💰',
        split: '🔄'
      };
      return icons[type] || '📊';
    },
    viewChart(holding) {
      this.$router.push(`/chart/${holding.symbol}`);
    },
    editPosition(holding) {
      console.log('Edit position:', holding.symbol);
    },
    sellPosition(holding) {
      console.log('Sell position:', holding.symbol);
    }
  },
  mounted() {
    this.fetchPortfolioData();
  }
}
</script>

<style scoped>
.portfolio {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  padding: var(--space-8) 0;
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

.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-6);
}

.summary-card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  display: flex;
  align-items: center;
  gap: var(--space-4);
  box-shadow: var(--shadow-sm);
  transition: all var(--transition-fast);
}

.summary-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.card-icon {
  font-size: 2.5rem;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--color-primary-light);
  border-radius: var(--radius-lg);
}

.card-content {
  flex: 1;
}

.card-title {
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  color: var(--color-text-secondary);
  margin: 0 0 var(--space-1) 0;
}

.card-value {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  color: var(--color-text-primary);
  margin: 0 0 var(--space-1) 0;
}

.card-subtitle {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin: 0;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-controls {
  display: flex;
  gap: var(--space-2);
}

.chart-container {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-12);
  box-shadow: var(--shadow-sm);
}

.chart-placeholder {
  text-align: center;
  color: var(--color-text-secondary);
}

.chart-icon {
  font-size: 4rem;
  margin-bottom: var(--space-4);
}

.table-container {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.holdings-table {
  width: 100%;
  border-collapse: collapse;
}

.holdings-table th {
  background-color: var(--gray-50);
  padding: var(--space-4) var(--space-6);
  text-align: left;
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

.dark .holdings-table th {
  background-color: var(--color-surface-elevated);
}

.holdings-table td {
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

.symbol-info {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.symbol-text {
  font-family: var(--font-family-mono);
  font-weight: var(--font-bold);
  color: var(--color-primary);
}

.company-name {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
}

.action-buttons {
  display: flex;
  gap: var(--space-2);
  justify-content: center;
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

.bottom-section {
  margin-top: var(--space-12);
}

.bottom-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-8);
}

.allocation-section,
.activity-section {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
}

.allocation-container {
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
}

.allocation-chart {
  display: flex;
  justify-content: center;
}

.chart-placeholder-small {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-text-secondary);
}

.pie-chart-icon {
  font-size: 3rem;
  margin-bottom: var(--space-2);
}

.allocation-legend {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: var(--radius-sm);
}

.legend-label {
  flex: 1;
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
}

.legend-value {
  font-weight: var(--font-bold);
  color: var(--color-text-primary);
}

.activity-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.activity-item {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  padding: var(--space-3);
  border-radius: var(--radius-md);
  transition: background-color var(--transition-fast);
}

.activity-item:hover {
  background-color: var(--gray-50);
}

.dark .activity-item:hover {
  background-color: var(--color-surface-elevated);
}

.activity-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-full);
  font-size: 1.2rem;
}

.activity-icon.buy {
  background-color: var(--color-success-light);
}

.activity-icon.sell {
  background-color: var(--color-danger-light);
}

.activity-icon.dividend {
  background-color: var(--color-warning-light);
}

.activity-content {
  flex: 1;
}

.activity-description {
  font-weight: var(--font-medium);
  color: var(--color-text-primary);
  margin: 0 0 var(--space-1) 0;
}

.activity-date {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin: 0;
}

.activity-amount {
  font-weight: var(--font-bold);
  font-family: var(--font-family-mono);
}

.activity-footer {
  margin-top: var(--space-6);
  text-align: center;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-modal);
}

.modal-content {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-xl);
  max-width: 500px;
  width: 90%;
  max-height: 90vh;
  overflow: hidden;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-6);
  border-bottom: 1px solid var(--color-border);
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--color-text-secondary);
  padding: var(--space-2);
}

.modal-body {
  padding: var(--space-6);
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-3);
  padding: var(--space-6);
  border-top: 1px solid var(--color-border);
}

/* Responsive Design */
@media (max-width: 1024px) {
  .bottom-grid {
    grid-template-columns: 1fr;
  }
  
  .chart-controls {
    flex-wrap: wrap;
  }
}

@media (max-width: 768px) {
  .summary-cards {
    grid-template-columns: 1fr;
  }
  
  .section-header {
    flex-direction: column;
    gap: var(--space-4);
    align-items: stretch;
  }
  
  .table-container {
    overflow-x: auto;
  }
  
  .holdings-table {
    min-width: 900px;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .allocation-container {
    align-items: center;
  }
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>