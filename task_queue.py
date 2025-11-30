#!/usr/bin/env python3
"""
Async Task Queue for Headspace
Handles background document processing with progress tracking
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a background processing task"""
    id: str
    task_type: str  # 'process_document', 'generate_embeddings', 'retag_document'
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


class TaskQueue:
    """
    Async task queue for background document processing

    Allows multiple documents to be processed in parallel without blocking API
    """

    def __init__(self, max_concurrent: int = 3):
        """
        Initialize task queue

        Args:
            max_concurrent: Maximum number of concurrent tasks
        """
        self.tasks: Dict[str, Task] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.workers: List[asyncio.Task] = []
        self.running = False

    def add_task(self, task_type: str, func: Callable, *args, **kwargs) -> str:
        """
        Add a task to the queue

        Args:
            task_type: Type of task
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:12]

        task = Task(
            id=task_id,
            task_type=task_type,
            metadata={
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
        )

        self.tasks[task_id] = task

        # Add to queue
        asyncio.create_task(self.queue.put((task_id, func, args, kwargs)))

        return task_id

    async def worker(self, worker_id: int):
        """Worker coroutine that processes tasks from queue"""
        while self.running:
            try:
                # Get task from queue with timeout
                task_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                task_id, func, args, kwargs = task_data

                task = self.tasks.get(task_id)
                if not task:
                    continue

                # Update status
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                task.message = f"Processing with worker {worker_id}..."

                print(f"Worker {worker_id}: Starting task {task_id} ({task.task_type})")

                try:
                    # Execute task
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, func, *args, **kwargs)

                    # Task succeeded
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.progress = 1.0
                    task.completed_at = datetime.now()
                    task.message = "Completed successfully"

                    duration = (task.completed_at - task.started_at).total_seconds()
                    print(f"Worker {worker_id}: Completed task {task_id} in {duration:.2f}s")

                except Exception as e:
                    # Task failed
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    task.message = f"Failed: {str(e)}"

                    print(f"Worker {worker_id}: Task {task_id} failed: {e}")

                finally:
                    self.queue.task_done()

            except asyncio.TimeoutError:
                # No tasks in queue, continue waiting
                continue
            except Exception as e:
                print(f"Worker {worker_id}: Unexpected error: {e}")

    async def start(self):
        """Start worker tasks"""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self.worker(i))
            self.workers.append(worker)

        print(f"✅ TaskQueue started with {self.max_concurrent} workers")

    async def stop(self):
        """Stop all workers"""
        self.running = False

        # Wait for workers to finish current tasks
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        print(f"TaskQueue stopped")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks"""
        return list(self.tasks.values())

    def get_pending_tasks(self) -> List[Task]:
        """Get tasks that are pending or running"""
        return [t for t in self.tasks.values()
                if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]

    def clear_completed(self):
        """Remove completed and failed tasks"""
        self.tasks = {
            tid: task for tid, task in self.tasks.items()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        }


# Global task queue instance
task_queue = TaskQueue(max_concurrent=2)  # Process 2 documents at a time


if __name__ == "__main__":
    # Test the task queue
    import time

    async def test_task(name: str, duration: float):
        """Test task that simulates work"""
        print(f"  Task {name} starting (will take {duration}s)")
        await asyncio.sleep(duration)
        print(f"  Task {name} completed")
        return f"Result from {name}"

    async def main():
        # Start queue
        await task_queue.start()

        # Add some tasks
        task1_id = task_queue.add_task("test", test_task, "Task 1", 2.0)
        task2_id = task_queue.add_task("test", test_task, "Task 2", 1.5)
        task3_id = task_queue.add_task("test", test_task, "Task 3", 1.0)

        print(f"Added 3 tasks: {task1_id}, {task2_id}, {task3_id}")

        # Wait for all tasks to complete
        await asyncio.sleep(5)

        # Check results
        for task_id in [task1_id, task2_id, task3_id]:
            task = task_queue.get_task(task_id)
            print(f"Task {task_id}: {task.status.value} - {task.result}")

        # Stop queue
        await task_queue.stop()

    # Run test
    asyncio.run(main())