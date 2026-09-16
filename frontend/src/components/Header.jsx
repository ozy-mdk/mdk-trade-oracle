import React from 'react';
import { Activity, Database, RefreshCw } from 'lucide-react';

export default function Header({ onRefresh, loading }) {
  return (
    <header className="app-header glass-card">
      <div className="logo-group">
        <div className="logo-icon">
          <Activity size={22} />
        </div>
        <div>
          <h1 className="logo-title">MDK Trading Oracle</h1>
          <p className="logo-subtitle">Institutional Order Flow & Candlestick Terminal</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div className="status-pill">
          <span className="status-dot"></span>
          <span>PostgreSQL 18 &amp; GraphQL Active</span>
        </div>

        <button
          onClick={onRefresh}
          disabled={loading}
          className="tf-btn"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 12px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--border-subtle)',
          }}
          title="Verileri Yenile"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          <span>Yenile</span>
        </button>
      </div>
    </header>
  );
}
