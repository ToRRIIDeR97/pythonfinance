<template>
  <div class="error-message" :class="errorClass">
    <div class="error-icon" v-if="showIcon">
      {{ getIcon() }}
    </div>
    <div class="error-content">
      <h3 v-if="title" class="error-title">{{ title }}</h3>
      <p class="error-description">{{ message }}</p>
      <div v-if="details" class="error-details">
        <button 
          @click="showDetails = !showDetails" 
          class="details-toggle"
          :aria-expanded="showDetails"
        >
          {{ showDetails ? 'Hide' : 'Show' }} Details
          <span class="toggle-icon" :class="{ 'rotated': showDetails }">▼</span>
        </button>
        <div v-if="showDetails" class="details-content">
          <pre>{{ details }}</pre>
        </div>
      </div>
      <div v-if="actions.length > 0" class="error-actions">
        <button 
          v-for="action in actions" 
          :key="action.label"
          @click="action.handler"
          :class="['btn', action.variant || 'btn-primary']"
        >
          {{ action.label }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ErrorMessage',
  props: {
    type: {
      type: String,
      default: 'error',
      validator: value => ['error', 'warning', 'info', 'network', 'auth', 'validation'].includes(value)
    },
    title: {
      type: String,
      default: ''
    },
    message: {
      type: String,
      required: true
    },
    details: {
      type: String,
      default: ''
    },
    showIcon: {
      type: Boolean,
      default: true
    },
    actions: {
      type: Array,
      default: () => []
    },
    centered: {
      type: Boolean,
      default: true
    },
    compact: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      showDetails: false
    };
  },
  computed: {
    errorClass() {
      return {
        [`error-${this.type}`]: true,
        'centered': this.centered,
        'compact': this.compact
      };
    }
  },
  methods: {
    getIcon() {
      const icons = {
        error: '⚠️',
        warning: '⚠️',
        info: 'ℹ️',
        network: '🌐',
        auth: '🔒',
        validation: '❌'
      };
      return icons[this.type] || '⚠️';
    }
  }
}
</script>

<style scoped>
.error-message {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
  display: flex;
  gap: var(--space-4);
  align-items: flex-start;
}

.error-message.centered {
  text-align: center;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

.error-message.compact {
  padding: var(--space-4);
  min-height: auto;
}

/* Type-specific styling */
.error-error {
  border-color: var(--color-danger-light);
  background-color: var(--color-danger-light);
}

.error-warning {
  border-color: var(--color-warning-light);
  background-color: var(--color-warning-light);
}

.error-info {
  border-color: var(--color-primary-light);
  background-color: var(--color-primary-light);
}

.error-network {
  border-color: var(--color-secondary-light);
  background-color: var(--color-secondary-light);
}

.error-auth {
  border-color: var(--color-warning-light);
  background-color: var(--color-warning-light);
}

.error-validation {
  border-color: var(--color-danger-light);
  background-color: var(--color-danger-light);
}

.error-icon {
  font-size: 2.5rem;
  flex-shrink: 0;
}

.error-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.error-title {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  margin: 0;
}

.error-description {
  font-size: var(--text-base);
  color: var(--color-text-primary);
  line-height: var(--leading-relaxed);
  margin: 0;
}

.error-details {
  margin-top: var(--space-2);
}

.details-toggle {
  background: none;
  border: none;
  color: var(--color-primary);
  cursor: pointer;
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) 0;
  transition: color var(--transition-fast);
}

.details-toggle:hover {
  color: var(--color-primary-dark);
}

.toggle-icon {
  transition: transform var(--transition-fast);
  font-size: var(--text-xs);
}

.toggle-icon.rotated {
  transform: rotate(180deg);
}

.details-content {
  margin-top: var(--space-3);
  padding: var(--space-4);
  background-color: var(--gray-50);
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
}

.dark .details-content {
  background-color: var(--gray-800);
}

.details-content pre {
  font-family: var(--font-family-mono);
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  line-height: var(--leading-relaxed);
}

.error-actions {
  display: flex;
  gap: var(--space-3);
  margin-top: var(--space-2);
}

.error-message.centered .error-actions {
  justify-content: center;
}

/* Compact mode adjustments */
.error-message.compact .error-icon {
  font-size: 1.5rem;
}

.error-message.compact .error-title {
  font-size: var(--text-lg);
}

.error-message.compact .error-description {
  font-size: var(--text-sm);
}

/* Responsive design */
@media (max-width: 768px) {
  .error-message {
    flex-direction: column;
    text-align: center;
  }
  
  .error-actions {
    flex-direction: column;
    align-items: stretch;
  }
  
  .error-message.centered .error-actions {
    align-items: center;
  }
}

/* Animation for details */
.details-content {
  animation: slideDown 0.2s ease-out;
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

/* Focus styles for accessibility */
.details-toggle:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}
</style>