from bitstream.config.base import BaseConfigModule
from bitstream.index import NodeIndex, Connect
from typing import List, Optional
from bitstream.bit import Bit
import numbers
import struct
from fractions import Fraction

class GAInportConfig(BaseConfigModule):
    """General Array inport configuration.
    
    Based on general_array.inport*.inportX:
    - mask(8) + src_id(3) + pingpong_en(1) + pingpong_last_index(4) + 
      fp16to32(1) + int32tofp(1) = 18 bits
    """
    FIELD_MAP = [
        ("mask", 8, lambda x: int("".join(str(v) for v in x[::-1]), 2) if isinstance(x, list) else x),
        ("src_id", 1),  # ga_inport_src_id
        ("pingpong_en", 1),  # ga_inport_pingpong_en
        ("pingpong_last_index", 4),  # ga_inport_pingpong_last_index
        ("nbr_enable", 1),  # ga_inport_nbr_enable
        ("fp16tofp32", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_inport_fp16to32
        ("bf16tofp32", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_inport_bf16to32
        ("int32tofp32", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_inport_int32tofp
        ("uint8tofp32", 1,lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_inport_uint8tofp
        ("uint8toint32", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_inport_uint8to32
    ]
    
    def __init__(self, inport_idx: int):
        super().__init__()
        self.inport_idx = inport_idx
        self.id: Optional[NodeIndex] = None
    
    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None
        self.mark_empty()
    
    def from_json(self, cfg: dict):
        """Load from general_array.inport.inportX"""
        cfg = cfg.get("general_array", cfg)
        cfg = cfg.get("inport", cfg)
        key = f"inport{self.inport_idx}"
        
        if key in cfg:
            inport_cfg = cfg[key]
            if inport_cfg:
                # self.id = NodeIndex(f"GA_INPORT.{key}")
                super().from_json(inport_cfg)
            else:
                self.set_empty()
        else:
            self.set_empty()

class GAOutportConfig(BaseConfigModule):
    """General Array outport configuration.
    
    Based on general_array.outport.ga_outport_*:
    - mask(8) + src_id(3) + fp32to16(1) + int32to8(1) = 13 bits
    """
    FIELD_MAP = [
        ("mask", 8, lambda x: int("".join(str(v) for v in x[::-1]), 2) if isinstance(x, list) else x),
        ("src_id", 1),  # ga_outport_src_id
        ("fp32tofp16", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_outport_fp32tofp16
        ("fp32tobf16", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_outport_fp32tobf16
        ("int32touint8", 1, lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else x)),  # ga_outport_int32to8
    ]
    
    def __init__(self):
        super().__init__()
        self.id: Optional[NodeIndex] = None
    
    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None
        self.mark_empty()
    
    def from_json(self, cfg: dict):
        """Load from general_array.outport"""
        cfg = cfg.get("general_array", cfg)
        cfg = cfg.get("outport", cfg)
        if cfg:
            # self.id = NodeIndex("GA_OUTPORT")
            super().from_json(cfg)
        else:
            self.set_empty()

class GAPEConfig(BaseConfigModule):
    """General Array PE configuration.
    
    Based on general_array.PE_array.PE**.inportX:
    - inport2: src_id(3) + keep_last_index(4) + mode(2) + constant(32) = 41 bits
    - inport1: src_id(3) + keep_last_index(4) + mode(2) + constant(32) = 41 bits
    - inport0: src_id(3) + keep_last_index(4) + mode(2) + constant(32) = 41 bits
    - alu_opcode(3) = 3 bits
    Total: 41*3 + 3 = 126 bits
    """
    FIELD_MAP = [
        # ALU opcode (3 bits)
        ("alu_opcode", 5, lambda x: GAPEConfig._encode_opcode(x)),
        ("transout_last_index", 4),
        
        # Port 2: src_id(3) + keep_last_index(4) + mode(2) + constant(32)
        ("inport2_src_id", 3),
        ("inport2_keep_last_index", 4),
        ("inport2_mode", 2, lambda x: x if isinstance(x, int) else (GAPEConfig.inport_mode_map().get(x, 0) if x is not None else 0)),
        
        # Port 1: src_id(3) + keep_last_index(4) + mode(2) + constant(32)
        ("inport1_src_id", 3),
        ("inport1_keep_last_index", 4),
        ("inport1_mode", 2, lambda x: x if isinstance(x, int) else (GAPEConfig.inport_mode_map().get(x, 0) if x is not None else 0)),
        
        # Port 0: src_id(3) + keep_last_index(4) + mode(2) + constant(32)
        ("inport0_src_id", 3),
        ("inport0_keep_last_index", 4),
        ("inport0_mode", 2, lambda x: x if isinstance(x, int) else (GAPEConfig.inport_mode_map().get(x, 0) if x is not None else 0)),
        
        ("_padding0", 4),  # Padding to align to byte boundary
        ("constant0", 32, lambda x: GAPEConfig._encode_constant(x)),
        
        ("_padding1", 4),  # Padding to align to byte boundary
        ("constant1", 32, lambda x: GAPEConfig._encode_constant(x)),
        
        ("_padding0", 4),  # Padding to align to byte boundary
        ("constant2", 32, lambda x: GAPEConfig._encode_constant(x)),
    ]

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

    @staticmethod
    def _encode_opcode(val):
        """Encode opcode and reject unknown symbolic names."""
        if val is None:
            return 0
        if isinstance(val, int):
            return val
        opcode = GAPEConfig.opcode_map().get(val)
        if opcode is None:
            raise ValueError(f"Unsupported GAPE opcode: {val}")
        return opcode
    
    @staticmethod
    def opcode_map():
        """Map string opcode names to integers. Also accepts integers directly."""
        return {
            "add": 0,
            "sub": 1,
            "mul": 2,
            "max": 3,
            "sum": 4,
            "summac": 5,
            "mac": 6,
            "int8_max": 11,
            "int32_sum": 12,
            "int32_sub": 13,
            "int32_mac": 14,
            "rec": 17,
            "sqrt": 18,
            "rec_sqrt": 20,
            "ex": 24,
        }
    
    @staticmethod
    def inport_mode_map():
        """Map inport modes to integers. Also accepts integers directly."""
        return {
            None: 0,
            "buffer": 1,
            "keep": 2,
            "constant": 3,
        }
    
    def __init__(self, name: str):
        """Initialize with PE name (e.g., 'PE00', 'PE12')"""
        super().__init__()
        self.name = name
        self.id: Optional[NodeIndex] = None
    
    def set_empty(self):
        """Set all fields to None so that to_bits produces zeros."""
        for field_info in self.FIELD_MAP:
            name = field_info[0]
            self.values[name] = None
        self.mark_empty()
    
    def from_json(self, cfg: dict):
        """Fill this PE config from JSON by looking up PE by name"""
        cfg = cfg.get("general_array", cfg)
        cfg = cfg.get("PE_array", cfg)
        
        if self.name in cfg:
            entry = cfg[self.name]
            
            # Check if this PE has configuration data
            if not entry or ('alu_opcode' not in entry and 'inport0' not in entry):
                # Empty PE configuration
                self.set_empty()
            else:
                # Assign NodeIndex only if PE has data
                self.id = NodeIndex(f'GA_PE.{self.name}')
                
                # Parse JSON format into FIELD_MAP format
                self.values['alu_opcode'] = entry.get('alu_opcode', 0)
                transout_last_index = entry.get('transout_last_index', 0)
                self.values['transout_last_index'] = 15 if transout_last_index is None else transout_last_index
                
                # Extract fields for each port (inport2, inport1, inport0)
                for i in range(3):
                    port = entry.get(f'inport{i}', {})
                    src_id = port.get('src_id')
                    # Special case: if src_id is the string "buffer", set to 0
                    if isinstance(src_id, str) and src_id.lower() == "buffer":
                        self.values[f'inport{i}_src_id'] = 0
                    # If src_id is a string (node name), create a Connect object
                    elif isinstance(src_id, str):
                        self.values[f'inport{i}_src_id'] = Connect(src_id, self.id)
                    elif src_id is None:
                        self.values[f'inport{i}_src_id'] = 0
                    else:
                        # Keep integer src_id as is for backward compatibility
                        self.values[f'inport{i}_src_id'] = src_id
                    
                    self.values[f'inport{i}_keep_last_index'] = port.get('keep_last_index', 0)
                    self.values[f'inport{i}_mode'] = port.get('mode', 0)
                    self.values[f'constant{i}'] = port.get('constant', 0)
        else:
            # No valid entry: treat as empty
            self.set_empty()
