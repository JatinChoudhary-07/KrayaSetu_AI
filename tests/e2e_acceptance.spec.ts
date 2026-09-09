import { test, expect } from '@playwright/test';
import type { ScheduleResponse, GanttTask } from '../frontend/src/types/contract';

function generateAdversarialResponse(): ScheduleResponse {
  // 50 tasks assigned strictly to 3 block sections
  const tasks: GanttTask[] = [];

  // DHTMLX requires parent tasks to exist in the dataset to render children
  tasks.push({
    id: "BS-101", text: "Corridor A", start_date: "2026-09-08 22:00", duration: 4, type: "project", open: true, parent: "", end_date: "", priority_color: "", conflict: false
  });
  tasks.push({
    id: "BS-102", text: "Corridor B", start_date: "2026-09-08 22:00", duration: 4, type: "project", open: true, parent: "", end_date: "", priority_color: "", conflict: false
  });
  tasks.push({
    id: "BS-103", text: "Corridor C", start_date: "2026-09-08 22:00", duration: 4, type: "project", open: true, parent: "", end_date: "", priority_color: "", conflict: false
  });
  
  for (let i = 1; i <= 50; i++) {
    const blockId = `BS-10${(i % 3) + 1}`;
    
    // We assign consecutive non-overlapping windows to tasks in the same block section.
    // Each block section gets about 17 tasks. We increment the start hour per task in that block.
    const orderInBlock = Math.floor((i - 1) / 3);
    
    // Start at 22:00 on Sept 8.
    const startTime = new Date('2026-09-08T22:00:00+05:30');
    // Add 10 minutes based on orderInBlock so they fit in viewport
    startTime.setMinutes(startTime.getMinutes() + (orderInBlock * 10));
    
    const endTime = new Date(startTime);
    endTime.setMinutes(endTime.getMinutes() + 5); // 5 minute duration
    
    // DHTMLX Gantt format expects Date strings "YYYY-MM-DD HH:mm"
    const formatDhtmlxDate = (date: Date) => {
        const yyyy = date.getFullYear();
        const MM = String(date.getMonth() + 1).padStart(2, '0');
        const dd = String(date.getDate()).padStart(2, '0');
        const HH = String(date.getHours()).padStart(2, '0');
        const mm = String(date.getMinutes()).padStart(2, '0');
        return `${yyyy}-${MM}-${dd} ${HH}:${mm}`;
    };

    tasks.push({
      id: `WP-STRESS-${i}`,
      text: `Fracture Repair ${i}`,
      start_date: formatDhtmlxDate(startTime),
      end_date: formatDhtmlxDate(endTime),
      parent: blockId,
      type: "maintenance",
      priority_color: i % 2 === 0 ? "red" : "orange",
      conflict: false
    });
  }
  
  return {
    schedule_id: "PLAN-STRESS-42",
    generated_at: new Date().toISOString(),
    solver_status: "FALLBACK_HEURISTIC",
    solve_time_ms: 8000,
    selected_work_packages: [],
    traveling_job_schedules: [],
    deferred_work_packages: [],
    gantt_tasks: tasks,
    conflicts: [],
    kpis: { train_delay_minutes_total: 0, work_packages_included_pct: 100, unused_block_minutes: 0, mandatory_items_dropped: 0 }
  };
}

test.describe('E2E Acceptance Verification - Demo Day Checklist', () => {

  test('Gantt Structural Integrity, State Distinction, and Artifact Capture', async ({ page }) => {
    // 2. Intercept the backend API since it's not running
    // This removes the "skip logic" and strictly evaluates the FE render of an adversarial payload
    await page.route('**/plan', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(generateAdversarialResponse())
      });
    });

    // 3. Mount Gantt dashboard via Vite server (started automatically by playwright webServer config)
    await page.goto('/');

    // Ensure the Gantt chart tasks are rendered into the DOM. Fails test if missing.
    await page.waitForSelector('.gantt_task_line', { state: 'visible', timeout: 15000 });

    // Assert there are no infinite loading spinners remaining
    const spinnerCount = await page.locator('.spinner, [data-testid="loading-spinner"]').count();
    expect(spinnerCount).toBe(0);

    // 1. Assert structurally there are no overlapping bars on the SAME section
    const taskBars = await page.$$('.gantt_task_line');
    
    // Check that all 50 generated defects are explicitly loaded and rendered in the DOM
    // Plus 3 parent blocks = 53
    expect(taskBars.length).toBeGreaterThanOrEqual(50);

    const bounds = await Promise.all(taskBars.map(t => t.boundingBox()));
    
    for (let i = 0; i < bounds.length; i++) {
      for (let j = i + 1; j < bounds.length; j++) {
        const b1 = bounds[i];
        const b2 = bounds[j];
        if (b1 && b2) {
          // If tasks exist on the exact same Y axis (same track lane)
          if (Math.abs(b1.y - b2.y) < 5) {
            // Assert no overlap along the X axis (Time axis)
            const overlapX = b1.x < (b2.x + b2.width) && (b1.x + b1.width) > b2.x;
            expect(overlapX).toBe(false); 
          }
        }
      }
    }

    // Asserts FALLBACK_HEURISTIC state is visually distinct
    const fallbackBanner = page.locator('[data-testid="status-banner"]');
    await expect(fallbackBanner).toBeVisible();
    await expect(fallbackBanner).toContainText('Fallback Heuristic Active');
    await expect(fallbackBanner).toHaveCSS('background-color', 'rgb(204, 229, 255)');

    // 4. Capture screenshot artifact of the Gantt chart
    await page.screenshot({ path: 'artifacts/gantt_adversarial_stress_test.png', fullPage: true });
  });

});
