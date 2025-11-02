<template>
  <div 
    :class="cardClass"
    @click="handleClick"
    :style="cardStyle"
  >
    <!-- Header -->
    <div v-if="showHeader" class="card-header" :class="headerClass">
      <div class="card-title-section">
        <h3 v-if="title" class="card-title">{{ title }}</h3>
        <p v-if="subtitle" class="card-subtitle">{{ subtitle }}</p>
        <slot name="title"></slot>
      </div>
      <div v-if="$slots.actions || actions.length > 0" class="card-actions">
        <slot name="actions">
          <button
            v-for="action in actions"
            :key="action.label"
            @click.stop="action.handler"
            :class="['btn', 'btn-sm', action.variant || 'btn-secondary']"
            :disabled="action.disabled"
          >
            {{ action.label }}
          </button>
        </slot>
      </div>
    </div>

    <!-- Media -->
    <div v-if="$slots.media" class="card-media">
      <slot name="media"></slot>
    </div>

    <!-- Body -->
    <div v-if="$slots.default || content" class="card-body" :class="bodyClass">
      <slot>
        <p v-if="content" class="card-content">{{ content }}</p>
      </slot>
    </div>

    <!-- Footer -->
    <div v-if="$slots.footer" class="card-footer" :class="footerClass">
      <slot name="footer"></slot>
    </div>

    <!-- Loading Overlay -->
    <div v-if="loading" class="card-loading">
      <LoadingSpinner size="medium" :message="loadingMessage" />
    </div>
  </div>
</template>

<script>
import LoadingSpinner from './LoadingSpinner.vue';

export default {
  name: 'Card',
  components: {
    LoadingSpinner
  },
  props: {
    title: {
      type: String,
      default: ''
    },
    subtitle: {
      type: String,
      default: ''
    },
    content: {
      type: String,
      default: ''
    },
    variant: {
      type: String,
      default: 'default',
      validator: value => ['default', 'outlined', 'elevated', 'filled', 'gradient'].includes(value)
    },
    size: {
      type: String,
      default: 'medium',
      validator: value => ['small', 'medium', 'large'].includes(value)
    },
    color: {
      type: String,
      default: 'default'
    },
    clickable: {
      type: Boolean,
      default: false
    },
    hoverable: {
      type: Boolean,
      default: true
    },
    loading: {
      type: Boolean,
      default: false
    },
    loadingMessage: {
      type: String,
      default: 'Loading...'
    },
    disabled: {
      type: Boolean,
      default: false
    },
    rounded: {
      type: Boolean,
      default: true
    },
    shadow: {
      type: Boolean,
      default: true
    },
    border: {
      type: Boolean,
      default: true
    },
    padding: {
      type: String,
      default: 'normal',
      validator: value => ['none', 'small', 'normal', 'large'].includes(value)
    },
    actions: {
      type: Array,
      default: () => []
    },
    width: {
      type: String,
      default: ''
    },
    height: {
      type: String,
      default: ''
    }
  },
  emits: ['click'],
  computed: {
    cardClass() {
      return [
        'card',
        `card-${this.variant}`,
        `card-${this.size}`,
        `card-padding-${this.padding}`,
        {
          [`card-${this.color}`]: this.color !== 'default',
          'card-clickable': this.clickable,
          'card-hoverable': this.hoverable && !this.disabled,
          'card-disabled': this.disabled,
          'card-loading': this.loading,
          'card-rounded': this.rounded,
          'card-shadow': this.shadow,
          'card-border': this.border
        }
      ];
    },
    cardStyle() {
      const style = {};
      if (this.width) style.width = this.width;
      if (this.height) style.height = this.height;
      return style;
    },
    showHeader() {
      return this.title || this.subtitle || this.$slots.title || this.$slots.actions || this.actions.length > 0;
    },
    headerClass() {
      return {
        'has-actions': this.$slots.actions || this.actions.length > 0
      };
    },
    bodyClass() {
      return {
        'no-padding': this.padding === 'none'
      };
    },
    footerClass() {
      return {};
    }
  },
  methods: {
    handleClick(event) {
      if (!this.disabled && this.clickable) {
        this.$emit('click', event);
      }
    }
  }
}
</script>

<style scoped>
.card {
  background-color: var(--color-surface);
  position: relative;
  overflow: hidden;
  transition: all var(--transition-fast);
  display: flex;
  flex-direction: column;
}

/* Variants */
.card-default {
  background-color: var(--color-surface);
}

.card-outlined {
  background-color: transparent;
  border: 2px solid var(--color-border);
}

.card-elevated {
  background-color: var(--color-surface);
  box-shadow: var(--shadow-lg);
}

.card-filled {
  background-color: var(--gray-50);
}

.dark .card-filled {
  background-color: var(--gray-800);
}

.card-gradient {
  background: linear-gradient(135deg, var(--color-primary-light), var(--color-primary));
  color: white;
}

/* Sizes */
.card-small {
  min-height: 120px;
}

.card-medium {
  min-height: 200px;
}

.card-large {
  min-height: 300px;
}

/* Colors */
.card-primary {
  border-color: var(--color-primary);
}

.card-success {
  border-color: var(--color-success);
}

.card-warning {
  border-color: var(--color-warning);
}

.card-danger {
  border-color: var(--color-danger);
}

/* States */
.card-clickable {
  cursor: pointer;
}

.card-hoverable:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.card-disabled {
  opacity: 0.6;
  cursor: not-allowed;
  pointer-events: none;
}

.card-loading {
  pointer-events: none;
}

/* Styling options */
.card-rounded {
  border-radius: var(--radius-lg);
}

.card-shadow {
  box-shadow: var(--shadow-sm);
}

.card-border {
  border: 1px solid var(--color-border);
}

/* Padding variants */
.card-padding-none .card-body {
  padding: 0;
}

.card-padding-small .card-header,
.card-padding-small .card-body,
.card-padding-small .card-footer {
  padding: var(--space-3);
}

.card-padding-normal .card-header,
.card-padding-normal .card-body,
.card-padding-normal .card-footer {
  padding: var(--space-4);
}

.card-padding-large .card-header,
.card-padding-large .card-body,
.card-padding-large .card-footer {
  padding: var(--space-6);
}

/* Header */
.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.card-header.has-actions {
  align-items: center;
}

.card-title-section {
  flex: 1;
  min-width: 0;
}

.card-title {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  margin: 0 0 var(--space-1) 0;
  line-height: var(--leading-tight);
}

.card-subtitle {
  font-size: var(--text-sm);
  color: var(--color-text-secondary);
  margin: 0;
  line-height: var(--leading-normal);
}

.card-actions {
  display: flex;
  gap: var(--space-2);
  flex-shrink: 0;
  margin-left: var(--space-4);
}

/* Media */
.card-media {
  flex-shrink: 0;
  overflow: hidden;
}

.card-media img {
  width: 100%;
  height: auto;
  display: block;
}

/* Body */
.card-body {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.card-content {
  color: var(--color-text-primary);
  line-height: var(--leading-relaxed);
  margin: 0;
}

/* Footer */
.card-footer {
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}

/* Loading overlay */
.card-loading::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 1;
}

.dark .card-loading::after {
  background-color: rgba(0, 0, 0, 0.8);
}

.card-loading .card-loading {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
}

/* Responsive design */
@media (max-width: 768px) {
  .card-header {
    flex-direction: column;
    align-items: stretch;
    gap: var(--space-3);
  }
  
  .card-header.has-actions {
    align-items: stretch;
  }
  
  .card-actions {
    margin-left: 0;
    justify-content: flex-end;
  }
  
  .card-title {
    font-size: var(--text-base);
  }
  
  .card-subtitle {
    font-size: var(--text-xs);
  }
}

/* Special card types */
.card.stat-card {
  text-align: center;
}

.card.stat-card .card-body {
  justify-content: center;
  align-items: center;
}

.card.metric-card .card-title {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  color: var(--color-primary);
}

.card.alert-card {
  border-left: 4px solid var(--color-primary);
}

.card.alert-card.card-success {
  border-left-color: var(--color-success);
}

.card.alert-card.card-warning {
  border-left-color: var(--color-warning);
}

.card.alert-card.card-danger {
  border-left-color: var(--color-danger);
}

/* Animation for loading state */
@keyframes cardPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.card-loading {
  animation: cardPulse 2s ease-in-out infinite;
}

/* Focus styles for accessibility */
.card-clickable:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .card {
    transition: none;
  }
  
  .card-hoverable:hover {
    transform: none;
  }
  
  .card-loading {
    animation: none;
  }
}
</style>