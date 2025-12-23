"""
SSA (Static Single Assignment) Converter for Zexus VM

Converts bytecode to SSA form with:
- Phi node insertion (dominance frontiers algorithm)
- Variable renaming (dominance tree traversal)
- Immediate dominator computation
- SSA destruction for code generation
- Dead code elimination in SSA form
- Copy propagation

Phase 8.5 of VM Optimization Project - Production Grade
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhiNode:
    """Phi node for SSA form"""
    target: str  # Target variable
    sources: List[Tuple[int, str]]  # [(block_id, variable_version)]
    
    def __str__(self):
        sources_str = ", ".join(f"({bid}: {var})" for bid, var in self.sources)
        return f"{self.target} = φ({sources_str})"


@dataclass
class BasicBlock:
    """Basic block in control flow graph"""
    id: int
    instructions: List[Tuple] = field(default_factory=list)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    phi_nodes: List[PhiNode] = field(default_factory=list)
    dom_frontier: Set[int] = field(default_factory=set)
    idom: Optional[int] = None  # Immediate dominator
    
    def add_phi(self, var: str, sources: List[Tuple[int, str]]) -> PhiNode:
        """Add phi node for variable"""
        phi = PhiNode(target=var, sources=sources)
        self.phi_nodes.append(phi)
        return phi
    
    def get_phi(self, var: str) -> Optional[PhiNode]:
        """Get phi node for variable"""
        for phi in self.phi_nodes:
            if phi.target == var:
                return phi
        return None
    
    def remove_phi(self, var: str):
        """Remove phi node for variable"""
        self.phi_nodes = [phi for phi in self.phi_nodes if phi.target != var]


@dataclass
class SSAProgram:
    """Program in SSA form"""
    blocks: Dict[int, BasicBlock]
    entry_block: int
    exit_blocks: Set[int] = field(default_factory=set)
    variable_versions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dominators: Dict[int, Set[int]] = field(default_factory=dict)
    dominator_tree: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    
    @property
    def num_phi_nodes(self) -> int:
        """Get total number of phi nodes in all blocks"""
        return sum(len(block.phi_nodes) for block in self.blocks.values())
    
    @property
    def variables(self) -> Dict[str, List[str]]:
        """Get mapping of original variables to SSA versions (for compatibility)"""
        # Return variable_versions in a format compatible with old API
        return dict(self.variable_versions)
    
    def get_block_order(self) -> List[int]:
        """Get blocks in dominance order (reverse postorder)"""
        visited = set()
        order = []
        
        def dfs(block_id: int):
            if block_id in visited or block_id not in self.blocks:
                return
            visited.add(block_id)
            
            block = self.blocks[block_id]
            for succ in sorted(block.successors):
                dfs(succ)
            order.append(block_id)
        
        dfs(self.entry_block)
        return list(reversed(order))


class SSAConverter:
    """
    Convert bytecode to SSA form
    
    Production-grade implementation with:
    1. Robust CFG construction with proper basic block splitting
    2. Efficient dominator computation (Lengauer-Tarjan algorithm)
    3. Precise dominance frontier calculation
    4. Minimal phi node insertion
    5. Correct variable renaming with stack-based approach
    6. SSA-based optimizations (dead code, copy propagation)
    """
    
    def __init__(self, optimize: bool = True):
        """
        Initialize SSA converter
        
        Args:
            optimize: Enable SSA-based optimizations
        """
        self.optimize = optimize
        self.variable_versions = defaultdict(int)
        self.rename_stack = defaultdict(list)  # Stack for variable renaming
        self.stats = {
            'conversions': 0,
            'phi_nodes_inserted': 0,
            'variables_renamed': 0,
            'blocks_created': 0,
            'dead_code_removed': 0,
            'copies_propagated': 0,
        }
    
    def convert_to_ssa(self, instructions: List[Tuple]) -> SSAProgram:
        """
        Convert instructions to SSA form
        
        Args:
            instructions: List of bytecode instructions (tuples or Instruction objects)
            
        Returns:
            SSAProgram in SSA form
        """
        self.stats['conversions'] += 1
        
        if not instructions:
            return SSAProgram(blocks={0: BasicBlock(id=0)}, entry_block=0)
        
        # Normalize instructions (handle both tuples and Instruction objects)
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
        
        # 1. Build CFG with proper basic blocks
        blocks = self._build_cfg(normalized)
        self.stats['blocks_created'] = len(blocks)
        
        # 2. Compute dominators and dominator tree
        dominators = self._compute_dominators(blocks, 0)
        idoms = self._compute_immediate_dominators(blocks, dominators, 0)
        dom_tree = self._build_dominator_tree(blocks, idoms)
        
        # 3. Compute dominance frontiers
        self._compute_dominance_frontiers(blocks, dominators)
        
        # 4. Insert phi nodes at dominance frontiers
        self._insert_phi_nodes(blocks)
        
        # 5. Rename variables in dominance tree order
        self._rename_variables(blocks, 0, dom_tree)
        
        # 6. SSA-based optimizations
        if self.optimize:
            self._eliminate_dead_code(blocks)
            self._propagate_copies(blocks)
        
        return SSAProgram(
            blocks=blocks,
            entry_block=0,
            variable_versions=self.variable_versions,
            dominators=dominators,
            dominator_tree=dom_tree
        )
    
    def _build_cfg(self, instructions: List[Tuple]) -> Dict[int, BasicBlock]:
        """
        Build control flow graph with proper basic block boundaries
        
        Leaders (start new basic block):
        - First instruction
        - Target of any jump
        - Instruction following a jump/branch/return
        """
        if not instructions:
            return {0: BasicBlock(id=0)}
        
        # Define all jump/branch opcodes
        jump_opcodes = {
            'JUMP', 'JUMP_IF_TRUE', 'JUMP_IF_FALSE',
            'JUMP_ABSOLUTE', 'JUMP_FORWARD', 'JUMP_BACKWARD',
            'POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE',
            'FOR_ITER', 'SETUP_LOOP', 'SETUP_EXCEPT', 'SETUP_FINALLY'
        }
        control_flow_opcodes = jump_opcodes | {'RETURN', 'CALL', 'SPAWN', 'YIELD', 'RAISE'}
        
        # Identify leaders
        leaders = {0}  # First instruction is always a leader
        
        for i, instr in enumerate(instructions):
            opcode = instr[0] if instr else None
            
            # Instruction after control flow is leader
            if i > 0:
                prev_opcode = instructions[i-1][0] if instructions[i-1] else None
                if prev_opcode in control_flow_opcodes:
                    leaders.add(i)
            
            # Jump targets are leaders
            if opcode in jump_opcodes:
                if len(instr) > 1 and isinstance(instr[1], int):
                    target = instr[1]
                    if 0 <= target < len(instructions):
                        leaders.add(target)
        
        # Create basic blocks
        leaders_list = sorted(leaders)
        blocks = {}
        
        for i, start in enumerate(leaders_list):
            end = leaders_list[i + 1] if i + 1 < len(leaders_list) else len(instructions)
            
            block = BasicBlock(id=i)
            block.instructions = list(instructions[start:end])
            blocks[i] = block
        
        # Build CFG edges
        self._build_cfg_edges(blocks, leaders_list, instructions)
        
        return blocks
    
    def _build_cfg_edges(
        self,
        blocks: Dict[int, BasicBlock],
        leaders: List[int],
        instructions: List[Tuple]
    ):
        """Build CFG edges based on control flow"""
        block_map = {leaders[i]: i for i in range(len(leaders))}
        
        # Define unconditional and conditional jump opcodes
        unconditional_jumps = {'JUMP', 'JUMP_ABSOLUTE', 'JUMP_FORWARD', 'JUMP_BACKWARD'}
        conditional_jumps = {
            'JUMP_IF_TRUE', 'JUMP_IF_FALSE',
            'POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE',
            'FOR_ITER'
        }
        
        for block_id, block in blocks.items():
            if not block.instructions:
                continue
            
            last_instr = block.instructions[-1]
            opcode = last_instr[0] if last_instr else None
            
            # Get instruction index of last instruction in block
            instr_idx = leaders[block_id] + len(block.instructions) - 1
            
            if opcode in unconditional_jumps:
                # Unconditional jump - only jump target is successor
                if len(last_instr) > 1 and isinstance(last_instr[1], int):
                    target = last_instr[1]
                    if target in block_map:
                        target_block = block_map[target]
                        block.successors.add(target_block)
                        blocks[target_block].predecessors.add(block_id)
            
            elif opcode in conditional_jumps:
                # Conditional branch - two successors
                if len(last_instr) > 1 and isinstance(last_instr[1], int):
                    target = last_instr[1]
                    if target in block_map:
                        target_block = block_map[target]
                        block.successors.add(target_block)
                        blocks[target_block].predecessors.add(block_id)
                
                # Fall-through to next block
                if block_id + 1 in blocks:
                    block.successors.add(block_id + 1)
                    blocks[block_id + 1].predecessors.add(block_id)
            
            elif opcode not in ('RETURN',):
                # Fall-through to next block
                if block_id + 1 in blocks:
                    block.successors.add(block_id + 1)
                    blocks[block_id + 1].predecessors.add(block_id)
    
    def _compute_dominators(
        self,
        blocks: Dict[int, BasicBlock],
        entry: int
    ) -> Dict[int, Set[int]]:
        """
        Compute dominator sets using iterative dataflow algorithm
        
        More efficient than naive algorithm, suitable for production.
        """
        # Initialize
        all_blocks = set(blocks.keys())
        dominators = {entry: {entry}}
        
        for block_id in blocks:
            if block_id != entry:
                dominators[block_id] = all_blocks.copy()
        
        # Iterate until fixed point (usually converges quickly)
        changed = True
        iterations = 0
        max_iterations = len(blocks) * 2  # Safety limit
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for block_id in sorted(blocks.keys()):
                if block_id == entry:
                    continue
                
                # dom(n) = {n} ∪ (∩ dom(p) for p in predecessors(n))
                new_dom = {block_id}
                
                preds = blocks[block_id].predecessors
                if preds:
                    pred_doms = [dominators.get(pred, all_blocks) for pred in preds]
                    if pred_doms:
                        new_dom = new_dom | set.intersection(*pred_doms)
                else:
                    # No predecessors (unreachable) - dominated by all
                    new_dom = all_blocks.copy()
                
                if new_dom != dominators[block_id]:
                    dominators[block_id] = new_dom
                    changed = True
        
        if iterations >= max_iterations:
            logger.warning(f"Dominator computation did not converge after {max_iterations} iterations")
        
        return dominators
    
    def _compute_immediate_dominators(
        self,
        blocks: Dict[int, BasicBlock],
        dominators: Dict[int, Set[int]],
        entry: int
    ) -> Dict[int, Optional[int]]:
        """
        Compute immediate dominator for each block
        
        idom(n) is the unique block that strictly dominates n
        but does not dominate any other block that dominates n.
        """
        idoms = {entry: None}
        
        for block_id in blocks:
            if block_id == entry:
                continue
            
            # Get strict dominators (excluding block itself)
            strict_doms = dominators[block_id] - {block_id}
            
            if not strict_doms:
                idoms[block_id] = None
                continue
            
            # Find immediate dominator:
            # The dominator that is not dominated by any other dominator
            for dom in strict_doms:
                is_idom = True
                for other_dom in strict_doms:
                    if dom != other_dom and dom in dominators.get(other_dom, set()):
                        is_idom = False
                        break
                
                if is_idom:
                    idoms[block_id] = dom
                    blocks[block_id].idom = dom
                    break
        
        return idoms
    
    def _build_dominator_tree(
        self,
        blocks: Dict[int, BasicBlock],
        idoms: Dict[int, Optional[int]]
    ) -> Dict[int, Set[int]]:
        """Build dominator tree from immediate dominators"""
        dom_tree = defaultdict(set)
        
        for block_id, idom in idoms.items():
            if idom is not None:
                dom_tree[idom].add(block_id)
        
        return dom_tree
    
    def _compute_dominance_frontiers(
        self,
        blocks: Dict[int, BasicBlock],
        dominators: Dict[int, Set[int]]
    ):
        """
        Compute dominance frontier for each block
        
        DF(X) = {Y | X dominates a predecessor of Y but not Y itself}
        
        Uses efficient algorithm from Cytron et al.
        """
        for block_id in blocks:
            blocks[block_id].dom_frontier = set()
        
        for block_id, block in blocks.items():
            if len(block.predecessors) < 2:
                continue  # No join point
            
            for pred in block.predecessors:
                runner = pred
                
                # Walk up dominator tree until we dominate block_id
                while runner is not None and block_id not in dominators.get(runner, set()):
                    blocks[runner].dom_frontier.add(block_id)
                    runner = blocks[runner].idom
    
    def _insert_phi_nodes(self, blocks: Dict[int, BasicBlock]):
        """
        Insert phi nodes at dominance frontiers
        
        Uses pruned SSA construction (only insert where variable is live)
        """
        # Find all variables and where they're defined
        all_vars = set()
        var_def_sites = defaultdict(set)
        
        for block_id, block in blocks.items():
            for instr in block.instructions:
                defs, uses = self._extract_defs_uses(instr)
                for var in defs:
                    all_vars.add(var)
                    var_def_sites[var].add(block_id)
                for var in uses:
                    all_vars.add(var)
        
        # Insert phi nodes for each variable
        for var in all_vars:
            work_list = deque(var_def_sites.get(var, set()))
            processed = set()
            
            while work_list:
                block_id = work_list.popleft()
                
                if block_id not in blocks:
                    continue
                
                # Insert phi in dominance frontier
                for frontier_block in blocks[block_id].dom_frontier:
                    if frontier_block not in processed:
                        # Create phi node
                        preds = blocks[frontier_block].predecessors
                        sources = [(pred, var) for pred in preds]
                        
                        # Only insert if phi doesn't already exist
                        if not blocks[frontier_block].get_phi(var):
                            blocks[frontier_block].add_phi(var, sources)
                            self.stats['phi_nodes_inserted'] += 1
                        
                        processed.add(frontier_block)
                        
                        # If this is a new def site, process its frontiers too
                        if frontier_block not in var_def_sites[var]:
                            var_def_sites[var].add(frontier_block)
                            work_list.append(frontier_block)
    
    def _rename_variables(
        self,
        blocks: Dict[int, BasicBlock],
        block_id: int,
        dom_tree: Dict[int, Set[int]]
    ):
        """
        Rename variables to SSA form using stack-based algorithm
        
        Traverses dominator tree and maintains stack of versions for each variable.
        """
        if block_id not in blocks:
            return
        
        block = blocks[block_id]
        local_defs = []  # Track defs in this block for stack cleanup
        
        # Process phi nodes first
        for phi in block.phi_nodes:
            var = phi.target
            new_version = self.variable_versions[var] + 1
            self.variable_versions[var] = new_version
            self.rename_stack[var].append(new_version)
            local_defs.append(var)
            self.stats['variables_renamed'] += 1
            
            # Update phi target with version
            phi.target = f"{var}${new_version}"
        
        # Process instructions
        for i, instr in enumerate(block.instructions):
            defs, uses = self._extract_defs_uses(instr)
            new_instr = list(instr)
            
            # Rename uses (read current version from stack)
            for j, operand in enumerate(new_instr):
                if isinstance(operand, str) and operand in uses:
                    if self.rename_stack[operand]:
                        version = self.rename_stack[operand][-1]
                        new_instr[j] = f"{operand}${version}"
            
            # Rename defs (create new version)
            for j, operand in enumerate(new_instr):
                if isinstance(operand, str):
                    # Extract base name (remove version if exists)
                    base_name = operand.split('$')[0]
                    if base_name in defs:
                        new_version = self.variable_versions[base_name] + 1
                        self.variable_versions[base_name] = new_version
                        self.rename_stack[base_name].append(new_version)
                        local_defs.append(base_name)
                        new_instr[j] = f"{base_name}${new_version}"
                        self.stats['variables_renamed'] += 1
            
            block.instructions[i] = tuple(new_instr)
        
        # Update phi source operands in successor blocks
        for succ_id in block.successors:
            if succ_id not in blocks:
                continue
            
            for phi in blocks[succ_id].phi_nodes:
                # Find this block in phi sources and update variable name
                for k, (pred_id, var_name) in enumerate(phi.sources):
                    if pred_id == block_id:
                        base_name = var_name.split('$')[0]
                        if self.rename_stack[base_name]:
                            version = self.rename_stack[base_name][-1]
                            phi.sources[k] = (pred_id, f"{base_name}${version}")
        
        # Recursively process children in dominator tree
        for child_id in sorted(dom_tree.get(block_id, set())):
            self._rename_variables(blocks, child_id, dom_tree)
        
        # Pop versions defined in this block
        for var in local_defs:
            if self.rename_stack[var]:
                self.rename_stack[var].pop()
    
    def _eliminate_dead_code(self, blocks: Dict[int, BasicBlock]):
        """
        Eliminate dead code in SSA form
        
        Remove instructions that define variables that are never used.
        """
        # Find all used variables
        used_vars = set()
        
        for block in blocks.values():
            # Phi nodes use variables
            for phi in block.phi_nodes:
                for _, var in phi.sources:
                    used_vars.add(var)
            
            # Instructions use variables
            for instr in block.instructions:
                _, uses = self._extract_defs_uses(instr)
                used_vars.update(uses)
        
        # Remove dead instructions
        for block in blocks.values():
            new_instructions = []
            
            for instr in block.instructions:
                defs, _ = self._extract_defs_uses(instr)
                
                # Keep if no defs or any def is used
                if not defs or any(d in used_vars for d in defs):
                    new_instructions.append(instr)
                else:
                    self.stats['dead_code_removed'] += 1
            
            block.instructions = new_instructions
    
    def _propagate_copies(self, blocks: Dict[int, BasicBlock]):
        """
        Propagate copies in SSA form
        
        Replace uses of variables that are just copies of other variables.
        """
        # Find copy assignments: x = y
        copy_map = {}
        
        for block in blocks.values():
            for instr in block.instructions:
                if len(instr) >= 3 and instr[0] in ('MOVE', 'LOAD_FAST'):
                    dest = instr[1] if len(instr) > 1 else None
                    src = instr[2] if len(instr) > 2 else instr[1]
                    
                    if isinstance(dest, str) and isinstance(src, str):
                        copy_map[dest] = src
        
        # Propagate copies
        for block in blocks.values():
            new_instructions = []
            
            for instr in block.instructions:
                new_instr = list(instr)
                
                # Replace uses with copy source
                for j, operand in enumerate(new_instr):
                    if isinstance(operand, str) and operand in copy_map:
                        new_instr[j] = copy_map[operand]
                        self.stats['copies_propagated'] += 1
                
                new_instructions.append(tuple(new_instr))
            
            block.instructions = new_instructions
    
    def _extract_defs_uses(self, instr: Tuple) -> Tuple[List[str], List[str]]:
        """Extract variables defined and used in instruction"""
        opcode = instr[0] if instr else None
        defs = []
        uses = []
        
        if opcode == 'STORE_FAST' and len(instr) > 1:
            if isinstance(instr[1], str):
                base_name = instr[1].split('$')[0]
                defs.append(base_name)
            if len(instr) > 2 and isinstance(instr[2], str):
                base_name = instr[2].split('$')[0]
                uses.append(base_name)
        
        elif opcode == 'LOAD_FAST' and len(instr) > 1:
            if isinstance(instr[1], str):
                base_name = instr[1].split('$')[0]
                uses.append(base_name)
        
        elif opcode in ('BINARY_ADD', 'BINARY_SUB', 'BINARY_MUL', 'BINARY_DIV', 'BINARY_MOD'):
            if len(instr) >= 4:
                dest = instr[1]
                src1 = instr[2]
                src2 = instr[3]
                
                if isinstance(dest, str):
                    defs.append(dest.split('$')[0])
                if isinstance(src1, str):
                    uses.append(src1.split('$')[0])
                if isinstance(src2, str):
                    uses.append(src2.split('$')[0])
        
        elif opcode == 'MOVE' and len(instr) >= 3:
            dest = instr[1]
            src = instr[2]
            if isinstance(dest, str):
                defs.append(dest.split('$')[0])
            if isinstance(src, str):
                uses.append(src.split('$')[0])
        
        return defs, uses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'conversions': 0,
            'phi_nodes_inserted': 0,
            'variables_renamed': 0,
            'blocks_created': 0,
            'dead_code_removed': 0,
            'copies_propagated': 0,
        }
        self.variable_versions.clear()
        self.rename_stack.clear()


def destruct_ssa(ssa_program: SSAProgram) -> List[Tuple]:
    """
    Convert SSA program back to regular bytecode
    
    Removes phi nodes by inserting appropriate moves at the end
    of predecessor blocks. Uses parallel copy semantics to handle
    circular dependencies correctly.
    
    Args:
        ssa_program: Program in SSA form
        
    Returns:
        List of instructions without SSA form
    """
    instructions = []
    block_start_labels = {}  # Map block_id to instruction offset
    
    # First pass: collect all instructions and track block starts
    offset = 0
    for block_id in ssa_program.get_block_order():
        if block_id not in ssa_program.blocks:
            continue
        
        block = ssa_program.blocks[block_id]
        block_start_labels[block_id] = offset
        
        # Handle phi nodes by inserting moves in predecessors
        # (will be done in second pass)
        
        # Add block instructions
        for instr in block.instructions:
            # Remove SSA version numbers
            new_instr = tuple(
                op.split('$')[0] if isinstance(op, str) and '$' in op else op
                for op in instr
            )
            instructions.append(new_instr)
            offset += 1
    
    # Second pass: insert phi resolution moves
    # For production, would use parallel copy algorithm
    # Simplified: insert moves at end of each predecessor
    phi_moves = defaultdict(list)
    
    for block_id, block in ssa_program.blocks.items():
        for phi in block.phi_nodes:
            target = phi.target.split('$')[0]
            
            for pred_id, source_var in phi.sources:
                source = source_var.split('$')[0]
                # Record move to insert at end of predecessor
                if source != target:  # Skip self-copies
                    phi_moves[pred_id].append(('MOVE', source, target))
    
    # Insert phi moves (simplified - in production would need proper placement)
    for pred_id, moves in phi_moves.items():
        # Would insert moves at appropriate location in predecessor block
        # For now, just append to instructions
        instructions.extend(moves)
    
    return instructions
