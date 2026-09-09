import React from 'react';
import type { SolverStatus } from './types/contract';

interface StatusBannerProps {
  status: SolverStatus;
}

export const StatusBanner: React.FC<StatusBannerProps> = ({ status }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'OPTIMAL':
        return { 
          backgroundColor: '#d4edda', 
          color: '#155724', 
          borderColor: '#c3e6cb',
          text: 'Optimal Plan Found' 
        };
      case 'FEASIBLE':
        return { 
          backgroundColor: '#fff3cd', 
          color: '#856404', 
          borderColor: '#ffeeba',
          text: 'Feasible Plan (Time Limit Reached)' 
        };
      case 'INFEASIBLE':
        return { 
          backgroundColor: '#f8d7da', 
          color: '#721c24', 
          borderColor: '#f5c6cb',
          text: 'Infeasible Plan (Cannot Satisfy Constraints)' 
        };
      case 'FALLBACK_HEURISTIC':
        return { 
          backgroundColor: '#cce5ff', 
          color: '#004085', 
          borderColor: '#b8daff',
          text: 'Fallback Heuristic Active (Solver Timed Out or Failed)' 
        };
      default:
        return { 
          backgroundColor: '#e2e3e5', 
          color: '#383d41', 
          borderColor: '#d6d8db',
          text: 'Unknown Status' 
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div 
      data-testid="status-banner"
      style={{
      padding: '12px 20px',
      backgroundColor: config.backgroundColor,
      color: config.color,
      border: `1px solid ${config.borderColor}`,
      fontWeight: 'bold',
      textAlign: 'center',
      borderRadius: '6px',
      margin: '10px 0',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <span>{config.text}</span>
      <span style={{ marginLeft: '10px', padding: '2px 8px', backgroundColor: 'rgba(0,0,0,0.1)', borderRadius: '4px', fontSize: '0.85em' }}>
        {status}
      </span>
    </div>
  );
};
