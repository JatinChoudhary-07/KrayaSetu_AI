import React from 'react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { StatusBanner } from './StatusBanner';
import { GanttDashboard } from './GanttDashboard';
import type { ScheduleResponse } from './types/contract';

const queryClient = new QueryClient();

function Dashboard() {
  const { data, isLoading, error } = useQuery<ScheduleResponse>({
    queryKey: ['plan'],
    queryFn: async () => {
      const res = await fetch('/plan', { method: 'POST', body: JSON.stringify({}) });
      if (!res.ok) throw new Error('Failed to fetch plan');
      return res.json();
    }
  });

  if (isLoading) return <div className="spinner" data-testid="loading-spinner">Loading...</div>;
  if (error) return <div>Error loading plan</div>;

  return (
    <div>
      <header style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '24px', margin: '0 0 10px 0' }}>SIH26027 Block Bundling Dashboard</h1>
        <StatusBanner status={data?.solver_status || ('UNKNOWN' as any)} />
      </header>
      <main>
        <div style={{ height: '3000px', border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
           {data?.gantt_tasks ? <GanttDashboard tasks={data.gantt_tasks} /> : <div>No tasks available</div>}
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
        <Dashboard />
      </div>
    </QueryClientProvider>
  );
}

export default App;
