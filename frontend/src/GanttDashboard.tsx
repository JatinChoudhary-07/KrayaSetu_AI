import React, { useEffect, useRef } from 'react';
import { gantt } from 'dhtmlx-gantt';
import 'dhtmlx-gantt/codebase/dhtmlxgantt.css';
import type { GanttTask } from './types/contract';

// Strictly typed payload for dispatcher overrides
export interface DispatcherOverridePayload {
  plan_id: string;
  override_type: 'LOCK_POSSESSION' | 'REJECT_CANDIDATE';
  block_id: string;
  reason: string;
  approved_by: string;
  approval_reference: string;
}

interface GanttDashboardProps {
  tasks: GanttTask[];
}

export const GanttDashboard: React.FC<GanttDashboardProps> = ({ tasks }) => {
  const ganttContainer = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ganttContainer.current) {
      // 1. Lanes & Hierarchy: Configured via tree columns
      // DHTMLX automatically creates lanes when tasks reference a `parent` ID
      gantt.config.columns = [
        { name: "text", label: "Task / Block Section", tree: true, width: 300, resize: true },
        { name: "start_date", label: "Start", align: "center", width: 120 },
        { name: "end_date", label: "End", align: "center", width: 120 }
      ];

      // 2. Visual Markers: Apply classes based on schema fields
      gantt.templates.task_class = (start, end, task: any) => {
        const classes = [];
        if (task.priority_color) {
          classes.push(`priority-${task.priority_color.toLowerCase()}`);
        }
        if (task.conflict) {
          classes.push('gantt-conflict-marker');
        }
        return classes.join(' ');
      };

      // 3. Community Edition Compliance
      // Explicitly avoid undo/redo, today marker, and multi-task drag plugins.
      gantt.config.drag_multiple = false;
      gantt.config.smart_rendering = false;
      
      // Standard scale configuration
      gantt.config.date_format = "%Y-%m-%d %H:%i";
      gantt.config.scale_unit = "hour";
      gantt.config.date_scale = "%H:%M";
      gantt.config.step = 1;
      gantt.config.min_column_width = 50;

      // Make the dashboard read-only as interactions should flow through typed hooks
      gantt.config.readonly = true;

      // Initialize the chart
      gantt.init(ganttContainer.current);
      
      // Parse data
      // The schema is strictly shaped field-for-field to match DHTMLX native format
      gantt.parse({ data: tasks, links: [] });
    }

    return () => {
      // Cleanup to prevent memory leaks on unmount
      gantt.clearAll();
    };
  }, [tasks]);

  // 4. Dispatcher Override Hooks
  // Must reject arbitrary free-form text by enforcing DispatcherOverridePayload type
  const applyDispatcherOverride = (payload: DispatcherOverridePayload): void => {
    console.log('Executing strictly typed dispatcher override:', payload);
    // TODO: dispatch to backend API
  };

  return (
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <div 
        ref={ganttContainer} 
        style={{ flex: 1, minHeight: '600px', width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid #e0e0e0' }} 
      />
      <style>{`
        /* Priority Lane Coloring */
        .gantt_task_line.priority-red {
          background-color: #ef4444 !important;
          border-color: #b91c1c !important;
        }
        .gantt_task_line.priority-orange {
          background-color: #f97316 !important;
          border-color: #c2410c !important;
        }
        .gantt_task_line.priority-yellow {
          background-color: #eab308 !important;
          border-color: #a16207 !important;
        }
        .gantt_task_line.priority-blue {
          background-color: #3b82f6 !important;
          border-color: #1d4ed8 !important;
        }
        .gantt_task_line.priority-green {
          background-color: #22c55e !important;
          border-color: #15803d !important;
        }
        
        /* Conflict Visual Marker */
        .gantt_task_line.gantt-conflict-marker {
          border: 2px dashed #000 !important;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.8) !important;
          background-image: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(0,0,0,0.1) 10px,
            rgba(0,0,0,0.1) 20px
          );
        }
      `}</style>
    </div>
  );
};
