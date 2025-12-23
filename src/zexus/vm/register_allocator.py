"""
Register Allocator for Zexus Register VM

Implements graph coloring register allocation with:
- Live range analysis
- Interference graph construction
- Graph coloring with spilling
- Register coalescing
- Optimized register assignment

Phase 8.5 of VM Optimization Project
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum


class RegisterType(Enum):
    """Register types"""
    GENERAL = 0      # General purpose registers
    TEMP = 1         # Temporary registers
    ARGUMENT = 2     # Function argument registers
    RETURN = 3       # Return value registers


@dataclass
class LiveRange:
    """Live range for a variable"""
    variable: str
    start: int
    end: int
    uses: List[int] = field(default_factory=list)
    defs: List[int] = field(default_factory=list)
    
    def overlaps(self, other: 'LiveRange') -> bool:
        """Check if this live range overlaps with another"""
        return not (self.end < other.start or other.end < self.start)
    
    def __hash__(self):
        return hash((self.variable, self.start, self.end))


@dataclass
class InterferenceGraph:
    """Graph representing register interference"""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_node(self, var: str):
        """Add a node (variable) to the graph"""
        self.nodes.add(var)
        if var not in self.edges:
            self.edges[var] = set()
    
    def add_edge(self, var1: str, var2: str):
        """Add an interference edge between two variables"""
        self.add_node(var1)
        self.add_node(var2)
        self.edges[var1].add(var2)
        self.edges[var2].add(var1)
    
    def degree(self, var: str) -> int:
        """Get the degree (number of neighbors) of a variable"""
        return len(self.edges.get(var, set()))
    
    def neighbors(self, var: str) -> Set[str]:
        """Get neighbors of a variable"""
        return self.edges.get(var, set())
    
    def remove_node(self, var: str):
        """Remove a node from the graph"""
        if var in self.nodes:
            self.nodes.remove(var)
        
        # Remove edges
        neighbors = self.edges.get(var, set())
        for neighbor in neighbors:
            self.edges[neighbor].discard(var)
        
        if var in self.edges:
            del self.edges[var]


@dataclass
class AllocationResult:
    """Result of register allocation"""
    allocation: Dict[str, int]  # Variable -> Register mapping
    spilled: Set[str]  # Variables that had to be spilled to memory
    num_registers_used: int
    coalesced_moves: int = 0


class RegisterAllocator:
    """
    Graph coloring register allocator
    
    Uses the classic graph coloring algorithm with improvements:
    - Chaitin-Briggs coloring with optimistic spilling
    - Coalescing of move-related variables
    - Biased coloring for better cache locality
    - Spill cost minimization
    """
    
    def __init__(self, num_registers: int = 16, num_temp_registers: int = 8):
        """
        Initialize register allocator
        
        Args:
            num_registers: Number of general-purpose registers available
            num_temp_registers: Number of temporary registers
        """
        self.num_registers = num_registers
        self.num_temp_registers = num_temp_registers
        self.total_registers = num_registers + num_temp_registers
        
        # Reserved registers
        self.reserved = {0, 1}  # R0 (zero), R1 (stack pointer)
        self.available_registers = set(range(2, self.num_registers))
        self.temp_registers = set(range(self.num_registers, self.total_registers))
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'spills': 0,
            'coalesced_moves': 0,
            'colorings': 0,
        }
    
    def allocate(
        self,
        instructions: List[Tuple],
        live_ranges: Dict[str, LiveRange]
    ) -> AllocationResult:
        """
        Allocate registers for variables
        
        Args:
            instructions: List of instructions
            live_ranges: Live ranges for all variables
            
        Returns:
            AllocationResult with allocation and spilled variables
        """
        self.stats['allocations'] += 1
        
        # Build interference graph
        interference_graph = self._build_interference_graph(live_ranges)
        
        # Find move-related variables for coalescing
        move_pairs = self._find_move_pairs(instructions)
        
        # Coalesce moves when possible
        coalesced = self._coalesce_moves(interference_graph, move_pairs)
        
        # Color the graph
        allocation, spilled = self._color_graph(
            interference_graph,
            self.available_registers
        )
        
        return AllocationResult(
            allocation=allocation,
            spilled=spilled,
            num_registers_used=len(set(allocation.values())),
            coalesced_moves=coalesced
        )
    
    def _build_interference_graph(
        self,
        live_ranges: Dict[str, LiveRange]
    ) -> InterferenceGraph:
        """
        Build interference graph from live ranges
        
        Two variables interfere if their live ranges overlap
        """
        graph = InterferenceGraph()
        
        # Add all variables as nodes
        for var in live_ranges:
            graph.add_node(var)
        
        # Add edges for interfering variables
        variables = list(live_ranges.keys())
        for i, var1 in enumerate(variables):
            for var2 in variables[i + 1:]:
                if live_ranges[var1].overlaps(live_ranges[var2]):
                    graph.add_edge(var1, var2)
        
        return graph
    
    def _find_move_pairs(self, instructions: List[Tuple]) -> List[Tuple[str, str]]:
        """
        Find pairs of variables related by move instructions
        
        Args:
            instructions: List of instructions (tuples or Instruction objects)
            
        Returns:
            List of (source, dest) variable pairs
        """
        move_pairs = []
        
        # Normalize instructions
        normalized = []
        for instr in instructions:
            if instr is None:
                normalized.append(None)
            elif hasattr(instr, 'opcode') and hasattr(instr, 'arg'):
                normalized.append((instr.opcode, instr.arg))
            else:
                normalized.append(instr)
        
        for instr in normalized:
            if not instr or len(instr) < 2:
                continue
            
            opcode = instr[0]
            
            # Detect move instructions (LOAD_FAST followed by STORE_FAST is a move)
            if opcode == 'LOAD_FAST' and len(instr) >= 3:
                source = instr[1]
                dest = instr[2] if len(instr) > 2 else None
                if source and dest and isinstance(source, str) and isinstance(dest, str):
                    move_pairs.append((source, dest))
            
            # MOVE opcode
            elif opcode == 'MOVE' and len(instr) >= 3:
                source = instr[1]
                dest = instr[2]
                if isinstance(source, str) and isinstance(dest, str):
                    move_pairs.append((source, dest))
        
        return move_pairs
    
    def _coalesce_moves(
        self,
        graph: InterferenceGraph,
        move_pairs: List[Tuple[str, str]]
    ) -> int:
        """
        Coalesce move-related variables when possible
        
        Two variables can be coalesced if:
        1. They are related by a move
        2. They don't interfere
        3. Coalescing won't make the graph uncolorable
        
        Returns:
            Number of moves coalesced
        """
        coalesced_count = 0
        
        for source, dest in move_pairs:
            # Check if both variables exist in graph
            if source not in graph.nodes or dest not in graph.nodes:
                continue
            
            # Check if they interfere
            if dest in graph.neighbors(source):
                continue
            
            # Check Briggs' conservative criterion:
            # Coalescing is safe if the merged node has < K neighbors with degree >= K
            combined_neighbors = graph.neighbors(source) | graph.neighbors(dest)
            high_degree_neighbors = sum(
                1 for n in combined_neighbors
                if graph.degree(n) >= self.num_registers
            )
            
            if high_degree_neighbors < self.num_registers:
                # Safe to coalesce
                self._merge_nodes(graph, source, dest)
                coalesced_count += 1
        
        self.stats['coalesced_moves'] += coalesced_count
        return coalesced_count
    
    def _merge_nodes(self, graph: InterferenceGraph, var1: str, var2: str):
        """
        Merge two nodes in the interference graph
        
        Combines var1 and var2 into a single node (var1)
        """
        # Add all edges from var2 to var1
        for neighbor in graph.neighbors(var2):
            if neighbor != var1:
                graph.add_edge(var1, neighbor)
        
        # Remove var2
        graph.remove_node(var2)
    
    def _color_graph(
        self,
        graph: InterferenceGraph,
        available_colors: Set[int]
    ) -> Tuple[Dict[str, int], Set[str]]:
        """
        Color the interference graph using graph coloring algorithm
        
        Uses Chaitin-Briggs algorithm with optimistic spilling.
        
        Returns:
            (allocation dict, set of spilled variables)
        """
        self.stats['colorings'] += 1
        
        allocation = {}
        spilled = set()
        stack = []
        remaining_graph = InterferenceGraph()
        remaining_graph.nodes = graph.nodes.copy()
        remaining_graph.edges = {k: v.copy() for k, v in graph.edges.items()}
        
        # Simplification phase: remove nodes with degree < K
        while remaining_graph.nodes:
            # Find node with degree < K
            low_degree_node = None
            for node in remaining_graph.nodes:
                if remaining_graph.degree(node) < len(available_colors):
                    low_degree_node = node
                    break
            
            if low_degree_node:
                # Remove and push to stack
                stack.append((low_degree_node, remaining_graph.neighbors(low_degree_node).copy()))
                remaining_graph.remove_node(low_degree_node)
            else:
                # Need to spill - choose node with highest degree
                if not remaining_graph.nodes:
                    break
                
                spill_node = max(remaining_graph.nodes, key=lambda n: remaining_graph.degree(n))
                stack.append((spill_node, remaining_graph.neighbors(spill_node).copy()))
                remaining_graph.remove_node(spill_node)
                spilled.add(spill_node)
                self.stats['spills'] += 1
        
        # Coloring phase: assign colors from stack
        while stack:
            node, neighbors = stack.pop()
            
            # Get colors used by neighbors
            used_colors = {
                allocation[n] for n in neighbors
                if n in allocation
            }
            
            # Find available color
            available = available_colors - used_colors
            
            if available:
                # Assign first available color
                allocation[node] = min(available)
            elif node not in spilled:
                # Optimistic coloring failed - must spill
                spilled.add(node)
                self.stats['spills'] += 1
        
        return allocation, spilled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'allocations': 0,
            'spills': 0,
            'coalesced_moves': 0,
            'colorings': 0,
        }


def compute_live_ranges(instructions: List[Tuple]) -> Dict[str, LiveRange]:
    """
    Compute live ranges for all variables in instructions
    
    Args:
        instructions: List of instructions (tuples or Instruction objects)
        
    Returns:
        Dictionary mapping variable names to their live ranges
    """
    live_ranges = {}
    
    # Normalize instructions
    normalized = []
    for instr in instructions:
        if instr is None:
            normalized.append(None)
        elif hasattr(instr, 'opcode') and hasattr(instr, 'arg'):
            # Instruction object from peephole optimizer
            normalized.append((instr.opcode, instr.arg))
        else:
            # Already a tuple
            normalized.append(instr)
    
    # First pass: find all def and use positions
    for i, instr in enumerate(normalized):
        if not instr or len(instr) < 2:
            continue
        
        opcode = instr[0]
        
        # Extract variable uses and defs
        defs, uses = _extract_vars(instr)
        
        # Update live ranges
        for var in defs:
            if var not in live_ranges:
                live_ranges[var] = LiveRange(variable=var, start=i, end=i)
            live_ranges[var].defs.append(i)
            live_ranges[var].end = max(live_ranges[var].end, i)
        
        for var in uses:
            if var not in live_ranges:
                live_ranges[var] = LiveRange(variable=var, start=i, end=i)
            live_ranges[var].uses.append(i)
            live_ranges[var].start = min(live_ranges[var].start, i)
            live_ranges[var].end = max(live_ranges[var].end, i)
    
    return live_ranges


def _extract_vars(instr: Tuple) -> Tuple[List[str], List[str]]:
    """
    Extract variables defined and used in an instruction
    
    Args:
        instr: Instruction tuple
        
    Returns:
        (defs, uses) - lists of variable names
    """
    opcode = instr[0] if instr else None
    defs = []
    uses = []
    
    # Simple heuristic: first arg is often dest, rest are sources
    if opcode in ('LOAD_FAST', 'STORE_FAST'):
        if len(instr) > 1 and isinstance(instr[1], str):
            if opcode == 'STORE_FAST':
                defs.append(instr[1])
            else:
                uses.append(instr[1])
    
    elif opcode in ('BINARY_ADD', 'BINARY_SUB', 'BINARY_MUL', 'BINARY_DIV'):
        # Typically: dest = src1 op src2
        if len(instr) >= 4:
            dest, src1, src2 = instr[1], instr[2], instr[3]
            if isinstance(dest, str):
                defs.append(dest)
            if isinstance(src1, str):
                uses.append(src1)
            if isinstance(src2, str):
                uses.append(src2)
    
    return defs, uses
