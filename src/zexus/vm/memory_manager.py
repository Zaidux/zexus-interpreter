"""
Zexus Memory Manager - Phase 7 Implementation

Provides memory management and garbage collection for the Zexus VM.

Features:
- Custom heap allocator with object pools
- Mark-and-sweep garbage collection
- Memory profiling and statistics
- Leak detection and prevention
- Configurable GC thresholds

Author: Zexus Team
Date: December 19, 2025
Version: 1.0
"""

import time
import weakref
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
import gc as system_gc


class ObjectState(Enum):
    """Object lifecycle states"""
    ALLOCATED = "allocated"
    MARKED = "marked"
    SWEPT = "swept"


@dataclass
class MemoryObject:
    """Managed memory object"""
    id: int
    obj: Any
    size: int
    allocated_at: float
    state: ObjectState = ObjectState.ALLOCATED
    ref_count: int = 0
    generation: int = 0
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return isinstance(other, MemoryObject) and self.id == other.id


@dataclass
class MemoryStats:
    """Memory statistics"""
    total_allocated: int = 0
    total_freed: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    gc_runs: int = 0
    gc_time: float = 0.0
    objects_collected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'total_allocated': f"{self.total_allocated:,} bytes",
            'total_freed': f"{self.total_freed:,} bytes",
            'current_usage': f"{self.current_usage:,} bytes",
            'peak_usage': f"{self.peak_usage:,} bytes",
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'gc_runs': self.gc_runs,
            'gc_time': f"{self.gc_time:.4f}s",
            'objects_collected': self.objects_collected,
            'memory_efficiency': f"{(self.total_freed / max(self.total_allocated, 1)) * 100:.1f}%"
        }


class Heap:
    """Custom heap allocator for Zexus VM"""
    
    def __init__(self, max_size: int = 100 * 1024 * 1024):  # 100MB default
        """Initialize heap allocator.
        
        Args:
            max_size: Maximum heap size in bytes
        """
        self.max_size = max_size
        self.objects: Dict[int, MemoryObject] = {}
        self.free_list: List[int] = []
        self.next_id = 0
        self.stats = MemoryStats()
        
    def allocate(self, obj: Any, size: Optional[int] = None) -> int:
        """Allocate memory for an object.
        
        Args:
            obj: Object to allocate memory for
            size: Size in bytes (None = auto-calculate)
        
        Returns:
            Object ID
        
        Raises:
            MemoryError: If allocation would exceed max heap size
        """
        if size is None:
            size = sys.getsizeof(obj)
        
        # Check heap limit
        if self.stats.current_usage + size > self.max_size:
            raise MemoryError(f"Heap full: {self.stats.current_usage + size} > {self.max_size}")
        
        # Reuse freed ID if available
        if self.free_list:
            obj_id = self.free_list.pop()
        else:
            obj_id = self.next_id
            self.next_id += 1
        
        # Create memory object
        mem_obj = MemoryObject(
            id=obj_id,
            obj=obj,
            size=size,
            allocated_at=time.time()
        )
        
        self.objects[obj_id] = mem_obj
        
        # Update statistics
        self.stats.total_allocated += size
        self.stats.current_usage += size
        self.stats.allocation_count += 1
        
        if self.stats.current_usage > self.stats.peak_usage:
            self.stats.peak_usage = self.stats.current_usage
        
        return obj_id
    
    def deallocate(self, obj_id: int) -> bool:
        """Deallocate memory for an object.
        
        Args:
            obj_id: Object ID to deallocate
        
        Returns:
            True if deallocated, False if not found
        """
        if obj_id not in self.objects:
            return False
        
        mem_obj = self.objects[obj_id]
        
        # Update statistics
        self.stats.total_freed += mem_obj.size
        self.stats.current_usage -= mem_obj.size
        self.stats.deallocation_count += 1
        
        # Remove object and add ID to free list
        del self.objects[obj_id]
        self.free_list.append(obj_id)
        
        return True
    
    def get_object(self, obj_id: int) -> Optional[Any]:
        """Get object by ID.
        
        Args:
            obj_id: Object ID
        
        Returns:
            Object or None if not found
        """
        mem_obj = self.objects.get(obj_id)
        return mem_obj.obj if mem_obj else None
    
    def get_memory_object(self, obj_id: int) -> Optional[MemoryObject]:
        """Get memory object wrapper by ID.
        
        Args:
            obj_id: Object ID
        
        Returns:
            MemoryObject or None if not found
        """
        return self.objects.get(obj_id)
    
    def get_all_objects(self) -> List[MemoryObject]:
        """Get all allocated objects.
        
        Returns:
            List of all memory objects
        """
        return list(self.objects.values())
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = MemoryStats()


class GarbageCollector:
    """Mark-and-sweep garbage collector"""
    
    def __init__(self, heap: Heap, threshold: int = 1000):
        """Initialize garbage collector.
        
        Args:
            heap: Heap allocator to manage
            threshold: Number of allocations before triggering GC
        """
        self.heap = heap
        self.threshold = threshold
        self.allocations_since_gc = 0
        self.root_set: Set[int] = set()
        
    def add_root(self, obj_id: int):
        """Add object to root set (prevents collection).
        
        Args:
            obj_id: Object ID to add to roots
        """
        self.root_set.add(obj_id)
    
    def remove_root(self, obj_id: int):
        """Remove object from root set.
        
        Args:
            obj_id: Object ID to remove from roots
        """
        self.root_set.discard(obj_id)
    
    def mark_phase(self) -> Set[int]:
        """Mark all reachable objects.
        
        Returns:
            Set of reachable object IDs
        """
        marked = set()
        work_list = list(self.root_set)
        
        while work_list:
            obj_id = work_list.pop()
            
            if obj_id in marked:
                continue
            
            marked.add(obj_id)
            
            # Mark the object
            mem_obj = self.heap.get_memory_object(obj_id)
            if mem_obj:
                mem_obj.state = ObjectState.MARKED
                
                # Find references to other managed objects
                # (In a real implementation, this would traverse object references)
                # For now, we assume objects with higher ref_count are referenced
        
        return marked
    
    def sweep_phase(self, marked: Set[int]) -> int:
        """Sweep (collect) unmarked objects.
        
        Args:
            marked: Set of marked object IDs
        
        Returns:
            Number of objects collected
        """
        collected = 0
        to_collect = []
        
        # Find objects to collect
        for obj_id, mem_obj in list(self.heap.objects.items()):
            if obj_id not in marked and obj_id not in self.root_set:
                to_collect.append(obj_id)
        
        # Collect objects
        for obj_id in to_collect:
            self.heap.deallocate(obj_id)
            collected += 1
        
        # Reset state for remaining objects
        for mem_obj in self.heap.get_all_objects():
            mem_obj.state = ObjectState.ALLOCATED
        
        return collected
    
    def collect(self, force: bool = False) -> Tuple[int, float]:
        """Run garbage collection cycle.
        
        Args:
            force: Force collection even if threshold not reached
        
        Returns:
            Tuple of (objects_collected, gc_time)
        """
        # Check if collection needed
        if not force and self.allocations_since_gc < self.threshold:
            return 0, 0.0
        
        start_time = time.time()
        
        # Mark phase
        marked = self.mark_phase()
        
        # Sweep phase
        collected = self.sweep_phase(marked)
        
        # Update statistics
        gc_time = time.time() - start_time
        self.heap.stats.gc_runs += 1
        self.heap.stats.gc_time += gc_time
        self.heap.stats.objects_collected += collected
        
        # Reset allocation counter
        self.allocations_since_gc = 0
        
        return collected, gc_time
    
    def should_collect(self) -> bool:
        """Check if garbage collection should run.
        
        Returns:
            True if GC should run
        """
        return self.allocations_since_gc >= self.threshold
    
    def record_allocation(self):
        """Record an allocation (for threshold tracking)"""
        self.allocations_since_gc += 1
        
        # Auto-trigger collection if threshold reached
        if self.should_collect():
            self.collect()


class MemoryManager:
    """Main memory manager for Zexus VM"""
    
    def __init__(
        self,
        max_heap_size: int = 100 * 1024 * 1024,
        gc_threshold: int = 1000,
        enable_profiling: bool = True
    ):
        """Initialize memory manager.
        
        Args:
            max_heap_size: Maximum heap size in bytes
            gc_threshold: Allocations before GC trigger
            enable_profiling: Enable memory profiling
        """
        self.heap = Heap(max_size=max_heap_size)
        self.gc = GarbageCollector(heap=self.heap, threshold=gc_threshold)
        self.enable_profiling = enable_profiling
        self.allocation_history: List[Tuple[float, int, int]] = []  # (time, obj_id, size)
        
    def allocate(self, obj: Any, root: bool = False) -> int:
        """Allocate memory for object.
        
        Args:
            obj: Object to allocate
            root: If True, add to root set (prevent collection)
        
        Returns:
            Object ID
        """
        obj_id = self.heap.allocate(obj)
        
        if root:
            self.gc.add_root(obj_id)
        
        # Record allocation for profiling
        if self.enable_profiling:
            mem_obj = self.heap.get_memory_object(obj_id)
            if mem_obj:
                self.allocation_history.append((time.time(), obj_id, mem_obj.size))
        
        # Record for GC
        self.gc.record_allocation()
        
        return obj_id
    
    def deallocate(self, obj_id: int) -> bool:
        """Deallocate object.
        
        Args:
            obj_id: Object ID
        
        Returns:
            True if deallocated
        """
        self.gc.remove_root(obj_id)
        return self.heap.deallocate(obj_id)
    
    def get(self, obj_id: int) -> Optional[Any]:
        """Get object by ID.
        
        Args:
            obj_id: Object ID
        
        Returns:
            Object or None
        """
        return self.heap.get_object(obj_id)
    
    def collect_garbage(self, force: bool = False) -> Tuple[int, float]:
        """Trigger garbage collection.
        
        Args:
            force: Force collection
        
        Returns:
            Tuple of (objects_collected, gc_time)
        """
        return self.gc.collect(force=force)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.heap.stats.to_dict()
    
    def get_memory_report(self) -> str:
        """Generate detailed memory report.
        
        Returns:
            Formatted memory report
        """
        stats = self.heap.stats
        report = []
        
        report.append("=" * 60)
        report.append("Zexus Memory Manager Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("Memory Usage:")
        report.append(f"  Current:     {stats.current_usage:,} bytes")
        report.append(f"  Peak:        {stats.peak_usage:,} bytes")
        report.append(f"  Total Alloc: {stats.total_allocated:,} bytes")
        report.append(f"  Total Freed: {stats.total_freed:,} bytes")
        report.append("")
        
        report.append("Allocation Statistics:")
        report.append(f"  Allocations:   {stats.allocation_count:,}")
        report.append(f"  Deallocations: {stats.deallocation_count:,}")
        report.append(f"  Live Objects:  {len(self.heap.objects):,}")
        report.append("")
        
        report.append("Garbage Collection:")
        report.append(f"  GC Runs:      {stats.gc_runs:,}")
        report.append(f"  GC Time:      {stats.gc_time:.4f}s")
        report.append(f"  Collected:    {stats.objects_collected:,} objects")
        if stats.gc_runs > 0:
            report.append(f"  Avg GC Time:  {stats.gc_time / stats.gc_runs:.4f}s")
        report.append("")
        
        efficiency = (stats.total_freed / max(stats.total_allocated, 1)) * 100
        report.append(f"Memory Efficiency: {efficiency:.1f}%")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def detect_leaks(self) -> List[MemoryObject]:
        """Detect potential memory leaks.
        
        Returns:
            List of objects that may be leaking
        """
        current_time = time.time()
        leak_threshold = 60.0  # Objects alive for 60+ seconds
        
        potential_leaks = []
        for mem_obj in self.heap.get_all_objects():
            age = current_time - mem_obj.allocated_at
            if age > leak_threshold and mem_obj.id not in self.gc.root_set:
                potential_leaks.append(mem_obj)
        
        return potential_leaks
    
    def reset(self):
        """Reset memory manager (for testing)"""
        self.heap = Heap(max_size=self.heap.max_size)
        self.gc = GarbageCollector(heap=self.heap, threshold=self.gc.threshold)
        self.allocation_history.clear()


# Convenience function for creating memory manager
def create_memory_manager(
    max_heap_mb: int = 100,
    gc_threshold: int = 1000
) -> MemoryManager:
    """Create a memory manager with common settings.
    
    Args:
        max_heap_mb: Maximum heap size in megabytes
        gc_threshold: Allocations before GC trigger
    
    Returns:
        Configured MemoryManager
    
    Example:
        >>> mm = create_memory_manager(max_heap_mb=50, gc_threshold=500)
        >>> obj_id = mm.allocate({"key": "value"})
        >>> obj = mm.get(obj_id)
    """
    max_heap_bytes = max_heap_mb * 1024 * 1024
    return MemoryManager(
        max_heap_size=max_heap_bytes,
        gc_threshold=gc_threshold,
        enable_profiling=True
    )
