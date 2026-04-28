from bitstream.config.base import BaseConfigModule
from bitstream.index import NodeIndex, Connect
from bitstream.config.mapper import NodeGraph
from typing import Optional, List
from bitstream.bit import Bit
import numbers
import struct
from fractions import Fraction

class DramLoopControlConfig(BaseConfigModule):
    """Represents a single DRAM loop configuration."""

    FIELD_MAP = [
        ("src_id", 4, lambda self, x: Connect(x, self.id) if x else None),  # source node ID, resolved later
        ("outmost_loop", 1),
        ("start", 17),  # initial_value in config
        ("stride", 17),
        ("end", 17),
        ("last_index", 4),
    ]

    def __init__(self, idx : int):
        super().__init__()
        self.idx = idx
        self.id : Optional[NodeIndex] = None
    
    @property
    def physical_index(self) -> int:
        """Get physical index from NodeIndex.physical_id."""
        if self.id is not None:
            return self.id.physical_id
        return self.idx  # Fallback to logical index

    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None  # None will encode as 0 in to_bits
        self.mark_empty()

    def from_json(self, cfg: dict):
        """
        Fill this loop control from JSON by picking the index-th entry
        that contains a 'stride' field.
        
        Args:
            cfg (dict): JSON dictionary containing loop configs.
        """
        cfg = cfg.get("dram_loop_configs", cfg)
        # Filter entries that contain 'stride', sorted to maintain order
        stride_entries = [(k, v) for k, v in sorted(cfg.items()) if "stride" in v]
        
        if self.idx < len(stride_entries):
            key, entry = stride_entries[self.idx]
            # Check if this entry has meaningful data (not just stride: 0)
            has_data = entry.get("stride", 0) != 0 or entry.get("src_id") is not None
            if has_data:
                self.id = NodeIndex("DRAM_LC." + key)
                # Physical index will be resolved automatically from NodeIndex.physical_id
                super().from_json(entry)
            else:
                # Empty configuration
                self.set_empty()
        else:
            # No valid entry: treat as empty
            self.set_empty()
            
class LCPEConfig(BaseConfigModule):
    """Configuration for a PE connected to loop controls (LC_PE)."""

    # Field order matches iga_pe.py: ALU_OPCODE | PORT2(SRC,KEEP,MODE) | PORT1 | PORT0 | CONST2 | CONST1 | CONST0
    # Total: 2 + 8*3 + 12*3 = 62 bits
    FIELD_MAP = [
        ("_padding", 16),
        # ALU opcode (2 bits)
        ("opcode", 2, lambda x: LCPEConfig.opcode_map()[x] if x is not None else 0),
        
        # Port 2: src_id(3) + keep_last_index(3) + mode(2) = 8 bits
        ("inport2_src", 4),
        ("inport2_last_index", 4),
        ("inport2_mode", 2, lambda x: LCPEConfig.inport_mode_map()[x] if x is not None else 0),
        
        # Port 1: src_id(3) + keep_last_index(3) + mode(2) = 8 bits
        ("inport1_src", 4),
        ("inport1_last_index", 4),
        ("inport1_mode", 2, lambda x: LCPEConfig.inport_mode_map()[x] if x is not None else 0),
        
        # Port 0: src_id(3) + keep_last_index(3) + mode(2) = 8 bits
        ("inport0_src", 4),
        ("inport0_last_index", 4),
        ("inport0_mode", 2, lambda x: LCPEConfig.inport_mode_map()[x] if x is not None else 0),
        
        # Constants: 3 × 12 bits = 36 bits
        ("constant2", 16, lambda x: LCPEConfig._encode_constant(x)),
        ("constant1", 16, lambda x: LCPEConfig._encode_constant(x)),
        ("constant0", 16, lambda x: LCPEConfig._encode_constant(x)),
    ]

    def __init__(self, idx: int):
        """
        Initialize PE config with a given index.
        The actual JSON entry will be picked in from_json() by index.
        """
        super().__init__()
        self.idx = idx
        self.id: Optional[NodeIndex] = None

    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None  # None will encode as 0 in to_bits
        self.mark_empty()

    @staticmethod
    def opcode_map():
        """Map string opcode names to integers. Also accepts integers directly."""
        return {
            "add": 0,
            "mul": 1,
            "mac": 2,
        }
        
    @staticmethod
    def inport_mode_map():
        """Map string inport modes to integers. Also accepts integers directly."""
        return {
            None: 0,
            "buffer": 1,
            "keep": 2,
            "constant": 3,
        }

    @staticmethod
    def encode_enable(lst: List[int]) -> int:
        if lst is None:
            return 0
        # Treat missing entries as 0
        padded = [(lst[i] if i < len(lst) and lst[i] else 0) for i in range(3)]
        return (padded[2] << 2) | (padded[1] << 1) | padded[0]
    
    @staticmethod
    def _encode_constant(val):
        """Encode constant to 32-bit int. Floats -> fp32 IEEE754, ints -> int."""
        if val is None:
            return 0
        if isinstance(val, str):
            text = val.strip()
            compact = text.replace(" ", "")
            # Try symbolic fraction first (e.g., "1.0 / 1024" or "1/1024"), then float.
            if "/" in compact:
                numerator_text, denominator_text = compact.split("/", maxsplit=1)
                try:
                    numerator = float(Fraction(numerator_text))
                    denominator = float(Fraction(denominator_text))
                    if denominator == 0:
                        return val
                    val = numerator / denominator
                except (ValueError, ZeroDivisionError):
                    pass
            if isinstance(val, str):
                try:
                    val = float(Fraction(text))
                except (ValueError, ZeroDivisionError):
                    try:
                        val = float(text)
                    except ValueError:
                        return val
        if isinstance(val, numbers.Integral):
            return int(val)
        if isinstance(val, numbers.Real):
            return int.from_bytes(struct.pack('<f', float(val)), byteorder='little', signed=False)
        return val

    def from_json(self, cfg: dict):
        """
        Fill this PE config from JSON by picking the index-th entry
        from lc_pe_configs.
        """
        cfg = cfg.get("lc_pe_configs", cfg)
        # Get all PE keys sorted (PE0, PE1, ...)
        keys = sorted(cfg.keys())

        if self.idx < len(keys):
            key = keys[self.idx]
            entry = cfg[key]
            
            # Check if this PE has configuration data
            if not entry or ('alu_opcode' not in entry and 'inport' not in entry):
                # Empty PE configuration
                self.set_empty()
            else:
                # Assign NodeIndex only if PE has data
                self.id = NodeIndex(f'LC_PE.{key}')
                # Parse JSON format into FIELD_MAP format
                # JSON has: {alu_opcode, inport: [{src_id, mode, keep_last_index, cfg_constant_pos}, ...]}
                
                self.values['opcode'] = entry.get('alu_opcode', 0)
                
                # Extract fields for each port (note: reversed order - port2, port1, port0)
                for i in range(3):
                    port = entry.get(f'inport{i}', {})
                    src_id = port.get('src_id')
                    # If src_id is a string (node name), create a Connect object
                    if isinstance(src_id, str):
                        self.values[f'inport{i}_src'] = Connect(src_id, self.id)
                    elif src_id is None:
                        self.values[f'inport{i}_src'] = 0
                    else:
                        # Keep integer src_id as is for backward compatibility
                        self.values[f'inport{i}_src'] = src_id
                    
                    self.values[f'inport{i}_last_index'] = port.get('keep_last_index', 0)
                    self.values[f'inport{i}_mode'] = port.get('mode', 0)
                    self.values[f'constant{i}'] = port.get('constant', 0)
        else:
            # No valid entry: treat as empty
            self.set_empty()

class BufferRowLCConfig(BaseConfigModule):
    """Represents a buffer row loop control configuration (ROW_LC)."""

    FIELD_MAP = [
        ("src_id", 4, lambda self, x: x if isinstance(x, int) else (Connect(x, self.id) if x else None)),
        ("start", 3),
        ("stride", 3),
        ("end", 3),
        ("last_index", 4),
    ]

    def __init__(self, group: str):
        super().__init__()
        self.group = group
        self.id: Optional[NodeIndex] = None

    def from_json(self, cfg: dict):
        self.id = NodeIndex(f"{self.group}.ROW_LC")
        cfg = cfg.get("ROW_LC", cfg)
        super().from_json(cfg)
        
    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None
        self.mark_empty()


class BufferColLCConfig(BaseConfigModule):
    """Represents a buffer column loop control configuration (COL_LC)."""

    FIELD_MAP = [
        ("src_id", 4, lambda self, x: x if isinstance(x, int) else (Connect(x, self.id) if x else None)),
        ("start", 6),
        ("stride", 6),
        ("end", 6),
        ("last_index", 4),
    ]

    def __init__(self, group: str):
        super().__init__()
        self.group = group
        self.id: Optional[NodeIndex] = None

    def from_json(self, cfg: dict):
        self.id = NodeIndex(f"{self.group}.COL_LC")
        cfg = cfg.get("COL_LC", cfg)
        super().from_json(cfg)
        
    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None
        self.mark_empty()
    
class BufferLoopControlGroupConfig(BaseConfigModule):
    """Group of buffer loop controls (row and column)."""

    def __init__(self, idx : int):
        super().__init__()
        self.idx = idx
        
    def from_json(self, cfg):
        """Fill this group config from JSON by picking the index-th group"""

        cfg = cfg.get("buffer_loop_configs", cfg)
        # Get all group keys (e.g., Group1, Group2, etc.)
        keys = list(cfg.keys())
        
        # Ensure the idx is within the range of available groups
        if self.idx < len(keys):
            # Get the group key corresponding to the idx
            key = keys[self.idx]
            cfg = cfg.get(key, cfg)
            
            # Get target from configuration and map to preferred index per new rule (single index per target)
            target = cfg.get("target", None)
            if target is not None:
                try:
                    target_idx_map = {
                        'A': 0,
                        'B': 1,
                        "B'": 2,
                        'C': 3,
                        'D': 4,
                    }
                    preferred = [target_idx_map.get(target)] if target in target_idx_map else []
                    node_graph = NodeGraph.get()
                    pool_len = len(node_graph.mapping.resource_pools.get("ROW_LC", []))
                    # Pick the first available index within pool, else skip assignment
                    chosen = next((i for i in preferred if i is not None and i < pool_len), None)
                    if chosen is not None:
                        node_graph.assign_node(key, f"GROUP{chosen}")
                        node_graph.assign_node(f"{key}.ROW_LC", f"ROW_LC{chosen}")
                        node_graph.assign_node(f"{key}.COL_LC", f"COL_LC{chosen}")
                except Exception:
                    # If target is not a single char or mapping fails, fallback: don't assign
                    pass
            
            # Check if this group has meaningful data (not just a comment)
            has_data = any(k in cfg for k in ["ROW_LC", "COL_LC"])
            
            if has_data:
                self.submodules = [BufferRowLCConfig(key), BufferColLCConfig(key)]
                super().from_json(cfg)
                
                # Initialize each submodule using the group configuration
                for submodule in self.submodules:
                    submodule.from_json(cfg)
            else:
                # Empty group
                self.submodules = [BufferRowLCConfig(""), BufferColLCConfig("")]
                self.set_empty()
        else:
            # If idx is out of range, treat it as an empty configuration
            self.submodules = [BufferRowLCConfig(""), BufferColLCConfig("")]
            self.set_empty()
            
    def to_bits(self) -> List[Bit]:
        """Concatenate all sub-config bitstreams in fixed order."""
        return sum((sub.to_bits() for sub in self.submodules), [])
    
    def set_empty(self):
        """Set all submodules to empty configurations."""
        for submodule in self.submodules:
            submodule.set_empty()
        self.mark_empty()