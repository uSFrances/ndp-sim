# 用于从json文件中具有自动地址规划功能并可以自动调整算子模板的算子execution plan生成器


## json文件读取
### json文件格式
* 顶层 `used_slices`
    * 28bit 二进制掩码字符串，例如 `0b1111`
    * 作为默认 slice 分配，若某个算子未单独声明 `used_slices`，则继承该默认值
* 算子编号
  * 算子类型(add/mul/matmul/max...)  
    * used_slices
        * 28bit 二进制掩码字符串，例如 `0b0000000000000000000000001111`
        * 每个算子单独维护，用于更灵活的 slice 分配
  * 输入A 
    * tensor形状(K * M * N)
    * 输入来源(来自其他算子(给出算子编号)/外部输入)
  * 输入B 
    * tensor形状(K * M * N)
    * 输入来源(来自其他算子(给出算子编号)/外部输入)
  * 输入C tensor形状
    * tensor形状(K * M * N)
    * 输入来源(来自其他算子(给出算子编号)/外部输入)
  * 输出D
    * tensor形状(K * M * N)
* 额外输入与扩展字段
        * 支持 `B'` 输入（适用于 `gemm_local` 等算子）；地址规划中 `B'` 与 `B` 共享同一地址空间。
        * 支持 tensor 级 `remapping` 字段（A/B/B'/C/D）：
                * 可为 `null`，或长度为 26 的数组。
                * 数组元素范围为 `0..25`。
        * `config_length` 语义为“该算子 config 数据包含多少个 64bit 行”。
    * `config_sfu` 语义为 SFU 配置类型（非布尔）：
        * 支持：`GELU`、`REC`、`REC_SQRT`、`ReLU`、`Sigmoid`、`SiLU`、`SQRT`、`Tanh`、`Ex`。
        * 兼容别名：`Sigmiod` 会自动归一为 `Sigmoid`。
        * 当 `config_sfu` 非空时，算子会额外生成一条 SFU 配置加载指令。

### 完成读取后
建立字表，根据读取顺序输出控制指令

## 指令生成

### 指令生成逻辑
单个算子可以被分割为四段指令：时钟使能指令、初始配置载入指令、关键寄存器修改指令(数据基地址修改(涉及地址规划)，由于算子计算形状修改导致的访存模块变量修改)、算子开始计算指令
* 时钟使能指令(通过广播的方式使能Slice的指定时钟, Clock_Enable)
  * 格式：{{29'd0}, {Clock_Select: 4bit(0b1111)}, {Slice_Mask: 28bit}, {Opcode: 3bit(Clock_Enable: 001)}}  
  * 该指令位于所有指令之前，一次运行只执行一次  
* 初始配置载入指令(每一类算子拥有同一种初始配置载入指令) (Load_Config)
    * 格式：{{Config_Length: 8bit}, {DDR_Config_Addr: 22bit}, {2'd0}, {Config_SFU: 1bit}, {Slice_Mask: 28bit}, {Opcode: 3bit(Config: 000)}}  
    * 变量解析：  
    Config_Length：配置长度，常量，参数化，每类算子值不同。  当前加速器设计的地址空间为32bit，按照字节编址。实际总线和DDR Bank的存储粒度为128bit，且实际使用的地址空间大小为8MB * 8 * 16，即占用0x4000_0000大小（30bit）的地址空间。因此综合考虑，只需要28bit的地址就可以完成整个空间的寻址。为了压缩该条指令，现要求Config配置包必须与BANK Row对齐（放在每行的起始处），从而省去6bit的Row寻址。因此最终使用22bit进行寻址。  
    DDR_Config_Addr:配置字在内存中的位置，常量，参数化，每类算子值不同。 
        config 起始地址强制落在 BANK Row 起点（`col=0, subword=0`），并按整行预留空间，保证下一包也从行首开始。
    Config_SFU：区分当前 Load_Config 的配置包类型。
        `0` 表示算子 bitstream 配置；`1` 表示 SFU 系数配置。
        当 `config_sfu` 非空时，单算子会生成两条 Load_Config：先下发 `Config_SFU=0` 的算子配置，再下发 `Config_SFU=1` 的 SFU 配置。
    Slice_Mask：控制哪些slice会接收到该指令，由当前算子的 `used_slices` 28bit 二进制掩码直接给出；若算子未单独配置，则继承顶层默认值。
    * 当前实现细节：
        * `DDR_Config_Addr` 由地址规划结果生成，不再回退模板静态地址字段。
        * 指令中的 `DDR_Config_Addr` 编码规则为：`ddr_config_addr_bin = full_addr >> 10`。
        * 当 `config_length=0` 时，`DDR_Config_Addr` 输出为 0。
        * 当 `config_length>0` 但缺失规划地址时，生成流程会报错终止。

* 关键寄存器修改指令 (Write_Reg, 为指定Slice单独写寄存器)
  该指令分为数据基地址寄存器修改与由于算子计算形状修改导致的访存模块变量修改。
  * 格式：{{Write_Value: 32bit}, {Write_Addr: 14bit}, {10'd0}, {Slice_ID: 5bit}, {Opcode: 3bit(WriteReg: 100)}}
  * 变量解析：  
  Write_Value: 指定该寄存器配置值  
  Write_Addr: 指定配置的寄存器地址  
  10'd0: 占位符  
  Slice_ID: 配置ID，指示该指令需要配置的单Slice  
  Opcode: 固定为3'b100。指示指令类型，代表该指令为Write_Reg  
  * 数据基地址修改  
  涉及地址规划
  * 由于算子计算形状修改导致的访存模块变量修改
  每类算子需要修改的寄存器不同，由于每个算子配置文件中包含一个初始size该size每个算子不同需要单独作为可以修改的常量进行储存，需要对初始size与json中读取到的目标size进行比较后再决定是否修改控制寄存器，若相同则不需要相关的修改控制寄存器指令
  * 要点，由于修改时会同时修改32位寄存器，但是往往修改寄存器时仅需修改个别位数，所以需要提前获知原有其他无关位数的原始值，避免对无关寄存器进行修改。
    * 当前实现已支持按 tensor 的 `remapping` 自动下发 `stream_engine.stream.address_remapping`（26 x 5bit 打包）。
  
* 算子开始计算指令 (Start_Comp)
  * 格式：{{33'd0}, {Slice_Mask: 28bit}, {Opcode: 3bit(Start:101)}}
  * 在每个算子载入与寄存器完成修改后输出该指令
  
### 储存地址生成
* 地址格式：{{slave:5bit},{bank:2bit},{row:13bit},{col:6bit},{subword:4}}  
* 芯片共有28个slices(参数化设置)，通过 parse 每个算子的 `used_slices` 掩码确定哪些 slice 被启用，地址中的 slave 用于区分 slice 的储存空间。  
* 地址分配策略：采用不释放内存策略，即新输入/输出数据数据自动往后排，占用新的地址。同一算子的统一数据的不同slices上的数据地址应只有slave不同。
* 当前实现补充规则：
    * 同一算子内 `B'` 复用 `B` 的地址分配结果（共址）。
    * config 数据也参与统一地址规划，并放在所有 tensor 数据之后。
    * config 分配前会对齐到新的 row 起始位置（`col=0`），并按整行预留。
    * 当算子存在 `config_sfu` 时，SFU 系数文件（`config/SFU_Coeff/<type>.txt`）也会参与地址规划，且独立于算子 config 分配地址。
    * `sca_cfg.json` 中地址显示为真实 byte 地址（不再右移 4bit）。
* 新增自动地址规划：输入/输出 tensor 按“只增不回收”方式顺序分配地址。
* 支持按 `used_slices` 掩码进行每算子独立 slice 规划，未配置时继承顶层默认掩码。
* 地址规划仍按 16Byte 对齐进行分配，但输出展示保持真实 byte 地址语义。
* 新增 config 统一地址规划能力：
    * config 数据纳入地址规划流程，不再依赖模板中的静态地址字段。
    * config 放置在所有数据 tensor 之后。
    * `config_length` 的单位为 64bit 行数，config 起始地址对齐到 BANK Row 起点。
    * `config_length=0` 时不分配 config 地址，`Load_Config` 地址字段置 0。
    * 当 `config_length>0` 但缺少规划地址时，生成流程会显式报错，避免静默回退。

* 新增 SFU 系数配置规划能力：
    * `config_sfu` 非空时，从 `config/SFU_Coeff/<type>.txt` 自动读取 SFU 配置行数（按 64bit 行计数）。
    * SFU 配置地址与算子 config 地址一样按 BANK Row 起点对齐并整行预留。
    * `sca_cfg.json` 中输出 `opX_sfu_config` 条目，路径位于 `install/cfg_pkg/<type>.txt`。
    * 输出阶段会自动复制对应 SFU 系数文件到 `install/cfg_pkg/`。


  
### 需要的修改寄存器获取
将会预先获得各个算子的一个变量修改寄存器变量计算函数，该函数输入算子的各个输入输出形状(A/B/C/D),输出需要修改的功能寄存器名称与对应的值的一个字表，后续在获取功能寄存器对应的地址后，根据该字表一一对应生成寄存器修改指令。  
寄存器更新不仅只包括储存地址更新，还包括一些控制寄存器更新，对于储存寄存器A/B/C/D分别对应表中的Read Memory Stream Engine0/Read Memory Stream Engine1/Read Memory Stream Engine2/Write Memory Stream Engine0。对于控制寄存器，需要修改的控制寄存器每个算子的情况不同需要通过一个函数来进行计算生成，目前该部分计算函数仍为占位；寄存器的名称与表1中命名规则相同，脚本已可先解析表1表2，得到各个寄存器对应的地址以及每个地址包含的寄存器字段关系，再根据需要修改的寄存器生成相关寄存器修改指令。

* 新增：逻辑编号到物理编号自动映射
    * `control_registers.py` 中各算子寄存器更新函数现在可以优先使用逻辑实例名（例如 `iga_lc0`、`rd_stream0`、`wr_stream0`），不再要求直接写物理编号。
    * 生成流程会在 `compute_control_register_updates` 阶段自动查找 `config/<op_type>/mapping_review.json`，并将逻辑实例名替换为物理实例名。
    * 映射来源为 `node_to_resource` 字段，当前已支持：
        * `DRAM_LC.LCx -> iga_lcx`
        * `GROUPx.ROW_LC -> iga_row_lcx`
        * `GROUPx.COL_LC -> iga_col_lcx`
        * `LC_PE.PEx -> iga_pex`
        * `STREAM.streamx -> rd_streamx/wr_streamx`（依据 `READ_STREAMn`/`WRITE_STREAMn`）
    * 若某算子目录下不存在 `mapping_review.json`，则保持旧行为（不做实例名替换）。
    * 当前用于调试的示例文件：`config/prefill_summac_fp32MN_fp32MN/mapping_review.json`。

`operator_base_info.json` 中的 `initial_size` 已切换为新格式（并已移除旧格式）：
* 新格式：`{"A": [K,M,N], "B": [K,M,N], "C": [K,M,N], "D": [K,M,N]}`（可按算子实际输入省略未使用项，但必须包含 `D`）
* 旧格式：`[K, M, N]` 已不再支持

### 寄存器地址获取
将会预先获知两个excel表格(csv格式)，表格1为每个模块的各种功能寄存器对应的位数与位数值，表格2而为各模块的各32位寄存器对应的一个地址。  

当前实现中的映射逻辑如下：
* 模块到实例前缀采用 `(大组名称, 模块名称)` 联合映射，而不是只看模块名称。这样可正确区分 `SA/GA` 中同名模块（如 `3*Inport`）。
* 字段位置不再依赖表1中的 `[high:low]` 范围；改为读取 `Xbit` 位宽并按同模块内行顺序自动累加定位。
* 对 `iga_pe` 与 `ga_pe` 的常量字段采用专用规则：
    * `iga_pe`：`cfg_constant_pos` 通过 `const0/1/2` 地址映射（`inport0/1/2 -> const0/1/2`）。
    * `ga_pe`：`inport*.constant` 通过 `const0/1/2` 地址映射（`inport0/1/2 -> const0/1/2`）。
* 常量字段不参与普通顺序累加（即从顺序中删除），因此普通字段如 `inport0.mode` 会从低位开始布局（例如 `[1:0]`）。

表1格式  (文件位置: ./config/register_map_with_groups1.csv)

<table>
    <tr>
        <td>大组名称</td>
        <td>模块名称</td>
        <td>字段</td>
        <td>配置名</td>
    </tr>
    <tr>
        <td>CONFIG</td>
        <td>CONFIG</td>
        <td>8bit[3:0]</td>
        <td>config[use+update]</td>
    </tr>
    <tr>
        <td rowspan="29" >IGA</td>
        <td rowspan="6">20 *DRAM LC</td>
        <td>4bit[47:44]</td>
        <td>dram_loop_configs.src_id</td>
    </tr>
    <tr>
        <td>1bit[43:43]</td>
        <td>dram_loop_configs.outmost_loop</td>
    </tr>
    <tr>
        <td>17bit[42:30]</td>
        <td>dram_loop_configs.start</td>
    </tr>
    <tr>
        <td>17bit[29:17]</td>
        <td>dram_loop_configs.stride</td>
    </tr>
    <tr>
        <td>17bit[16:4]</td>
        <td>dram_loop_configs.end</td>
    </tr>
    <tr>
        <td>4bit[3:0]</td>
        <td>dram_loop_configs.last_index</td>
    </tr>
    <tr>
        <td rowspan="5">5*BUFFER ROW LC</td>
        <td>4bit[16:13]</td>
        <td>buffer_loop_configs.ROW_LC.src_id</td>
    </tr>
    <tr>
        <td>3bit[12:10]</td>
        <td>buffer_loop_configs.ROW_LC.start</td>
    </tr>
    <tr>
        <td>3bit[9:7]</td>
        <td>buffer_loop_configs.ROW_LC.stride</td>
    </tr>
    <tr>
        <td>3bit[6:4]</td>
        <td>buffer_loop_configs.ROW_LC.end</td>
    </tr>
    <tr>
        <td>4bit[3:0]</td>
        <td>buffer_loop_configs.ROW_LC.last_index</td>
    </tr>
    <tr>
        <td rowspan="5">5*BUFFER COL LC</td>
        <td>4bit[25:22]</td>
        <td>buffer_loop_configs.COL_LC.src_id</td>
    </tr>
    <tr>
        <td>6bit[21:16]</td>
        <td>buffer_loop_configs.COL_LC.start</td>
    </tr>
    <tr>
        <td>6bit[15:10]</td>
        <td>buffer_loop_configs.COL_LC.stride</td>
    </tr>
    <tr>
        <td>6bit[9:4]</td>
        <td>buffer_loop_configs.COL_LC.end</td>
    </tr>
    <tr>
        <td>4bit[3:0]</td>
        <td>buffer_loop_configs.COL_LC.last_index</td>
    </tr>
    <tr>
        <td rowspan="13">10*LC PE</td>
        <td>2bit[67:66]</td>
        <td>lc_pe_configs.alu_opcode</td>
    </tr>
    <tr>
        <td>4bit[65:62]</td>
        <td>lc_pe_configs.inport2.src_id</td>
    </tr>
    <tr>
        <td>4bit[61:58]</td>
        <td>lc_pe_configs.inport2.keep_last_index</td>
    </tr>
    <tr>
        <td>2bit[57:56]</td>
        <td>lc_pe_configs.inport2.mode</td>
    </tr>
    <tr>
        <td>4bit[55:52]</td>
        <td>lc_pe_configs.inport1.src_id</td>
    </tr>
    <tr>
        <td>4bit[51:48]</td>
        <td>lc_pe_configs.inport1.keep_last_index</td>
    </tr>
    <tr>
        <td>2bit[47:46]</td>
        <td>lc_pe_configs.inport1.mode</td>
    </tr>
    <tr>
        <td>4bit[45:42]</td>
        <td>lc_pe_configs.inport0.src_id</td>
    </tr>
    <tr>
        <td>4bit[41:38]</td>
        <td>lc_pe_configs.inport0.keep_last_index</td>
    </tr>
    <tr>
        <td>2bit[37:36]</td>
        <td>lc_pe_configs.inport0.mode</td>
    </tr>
    <tr>
        <td>16bit[35:24]</td>
        <td>lc_pe_configs.inport2.cfg_constant_pos</td>
    </tr>
    <tr>
        <td>16bit[23:12]</td>
        <td>lc_pe_configs.inport1.cfg_constant_pos</td>
    </tr>
    <tr>
        <td>16bit[11:0]</td>
        <td>lc_pe_configs.inport0.cfg_constant_pos</td>
    </tr>
</table>



表2格式(文件位置: ./config/config_output.csv)  
<table>
    <tr>
        <td>模块名</td>
        <td>实例名</td>
        <td>寄存器地址</td>
        <td>寄存器位宽范围</td>
    </tr>
    <tr>
        <td rowspan="40">iga_lc</td>
        <td rowspan="2">iga_lc0</td>
        <td>000000000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000000000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc1</td>
        <td>000000100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000000100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc2</td>
        <td>000001000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000001000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc3</td>
        <td>000001100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000001100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc4</td>
        <td>000010000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000010000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc5</td>
        <td>000010100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000010100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc6</td>
        <td>000011000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000011000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc7</td>
        <td>000011100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000011100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc8</td>
        <td>000100000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000100000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc9</td>
        <td>000100100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000100100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc10</td>
        <td>000101000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000101000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc11</td>
        <td>000101100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000101100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc12</td>
        <td>000110000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000110000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc13</td>
        <td>000110100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000110100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc14</td>
        <td>000111000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000111000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc15</td>
        <td>000111100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>000111100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc16</td>
        <td>001000000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>001000000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc17</td>
        <td>001000100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>001000100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc18</td>
        <td>001001000000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>001001000001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="2">iga_lc19</td>
        <td>001001100000</td>
        <td>[29:0]</td>
    </tr>
    <tr>
        <td>001001100001</td>
        <td>[59:30]</td>
    </tr>
    <tr>
        <td rowspan="5">iga_row_lc</td>
        <td>iga_row_lc0</td>
        <td>001010000000</td>
        <td>[16:0]</td>
    </tr>
    <tr>
        <td>iga_row_lc1</td>
        <td>001010100000</td>
        <td>[16:0]</td>
    </tr>
    <tr>
        <td>iga_row_lc2</td>
        <td>001011000000</td>
        <td>[16:0]</td>
    </tr>
    <tr>
        <td>iga_row_lc3</td>
        <td>001011100000</td>
        <td>[16:0]</td>
    </tr>
    <tr>
        <td>iga_row_lc4</td>
        <td>001100000000</td>
        <td>[16:0]</td>
    </tr>
    <tr>
        <td rowspan="5">iga_col_lc</td>
        <td>iga_col_lc0</td>
        <td>001100100000</td>
        <td>[25:0]</td>
    </tr>
    <tr>
        <td>iga_col_lc1</td>
        <td>001101000000</td>
        <td>[25:0]</td>
    </tr>
    <tr>
        <td>iga_col_lc2</td>
        <td>001101100000</td>
        <td>[25:0]</td>
    </tr>
    <tr>
        <td>iga_col_lc3</td>
        <td>001110000000</td>
        <td>[25:0]</td>
    </tr>
    <tr>
        <td>iga_col_lc4</td>
        <td>001110100000</td>
        <td>[25:0]</td>
    </tr>
</table>


### 原有寄存器变量获取
将会预先获知每个算子的初始配置载入的对应码流，需要对该码流进行解析，还原出原有寄存器的值，该码流输出格式与寄存器地址获取过程中的表1对应的位数顺序进行输入，但是该码流中由于需要将各个模块的寄存器载入码流分开，分别对各个模块设置了chunk次数与padding0的位数，将码流分为了一个个chunk_size的输入，trunk与padding为用户预先输入的参数化常量，所以需要对码流文件进行解码后才能得到寄存器原始值。
* 原有寄存器码流文件示例：(文件储存地址示例: ./config/max_config_32_32_out/parsed_bitstream.txt ,max_config_32_32_out代表对应的算子类型，max表示算子为max算子，每个文件夹中存储码流的文件都被命名为parsed_bitstream.txt)  
iga_lc:  
0  
1 100000000000000000000000000000000000001000000000001000000001  
0  
1 000010000000000000000000000000000000001000000000000001000000  
0  
1 010100000000000000000000000000000000001000000000000000010001  
1 011000000000000000000000000000000000001000000000000000010010  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
  
iga_row_lc:  
1 00110000010010010  
0  
0  
0  
1 00000000010010011  
  
iga_col_lc:  
1 11000000000100001000000011  
0  
0  
0  
1 11000000000100001000000100  
  
iga_pe:  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
1 000000000000000001000000000000000000110000000001  
1 000000000000000000000000000000010000000000000000  
0  
0  
0  
0  
0  
0  

se_rd_mse:  
1 0110000111000100000001100101000000000000000000000000000001  
1 0001101110000000000000000000000000000000000000011111000000  
1 0000000000101101000001000000000000000000010000000000000010  
1 0000000000000000000000000000011001110001011110110101011010  
1 0100111001010001100000111101110011010110001011010100100101  
1 0000011100110001010010000011000100000100000000000000000000  
1 0000000000000000000000000000000000000000000000000000000000  
1 0000000000000000000000000000000000000000000000000000000000  
1 0000000000000000000000000000111101110011010110001011010100  
1 1001010000011100110001010010000011000100000100000100000010  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
0  
* 每个模块对应的chunk与chunk_size与padding信息如下
padding		chunk_size	chunk  
1bit	rd_stream	58	10  
4bit	wr_stream	62	8  
    special_array	32	1  
    buffer_manager_cluster	21	1  
    iga_pe	48	2  
    iga_row_lc	17	1  
    iga_col_lc	26	1  
    iga_lc	60	1  
    se_nse	7	1  
    ga_inport_group	20	1  
    ga_outport_group	12	1  
    ga_pe	36	4  
其中chunk_size代表一次chunk会发送的比特数，对应原寄存器码流文件中的一行的数据长度，chunk的大小代表一个模块完成寄存器写需要的次数，对应行数，每一行前面的0代表该模块是否被使用并被写存器，由此可以判断对应的模块编号，padding为对原内容进行padding后再进行chunk分割

## 当前实现状态
### 已完成
* 已完成 JSON 输入解析与校验，支持按算子顺序读取输入输出关系并生成内部字表。
* 已完成扩展输入与字段解析：支持 `B'` 与 tensor `remapping` 字段。
* 已完成自动地址规划，采用“不释放内存、顺序向后分配”的策略；同一 tensor 在不同 slice 上仅 slave 字段不同。
* 已完成 `B/B'` 共址分配规则与 config 统一地址规划（按 `config_length` 的 64bit 行数分配，config 起始地址与 BANK Row 对齐，并排在数据之后）。
* 已完成三段指令生成：Load_Config、Write_Reg、Start_Comp。
* 已完成全局单次 `Clock_Enable` 下发：一次运行仅在最前面发送一次。
* 已完成基地址寄存器写入：
    * 输入 A/B/C 分别映射到 `rd_stream0/rd_stream1/rd_stream2`
    * 输出 D 映射到 `wr_stream0`
    * 输入与输出 D 的 `base_addr` 均强制使用地址规划结果（不回退模板中的静态 base_addr）。
* 已完成 `Load_Config` 地址来源切换：仅使用地址规划结果，不再依赖模板中的静态 `ddr_config_addr/config_bitstream_addr` 地址信息。
* 已完成 SFU 双配置加载链路：当算子 `config_sfu` 非空时自动追加第二条 `Load_Config`（`Config_SFU=1`），并从 `config/SFU_Coeff` 自动读取对应系数文件长度。
* 已完成寄存器映射解析：通过 `config/register_map_with_groups1.csv` 与 `config/config_output.csv` 联合建立“字段 -> 地址位段”关系。
* 已完成原始寄存器值恢复：支持从 `parsed_bitstream.txt` 按模块分段读取原始码流。
* 已完成前置 padding 解码规则：当模块定义了 `padding_bits` 时，检索寄存器位段会按“padding 加在码流最开始”进行偏移。
* 已完成码流启用状态校验：若计划修改的寄存器地址对应源码流中未启用的 chunk（即该 chunk 首位为 `0`），会直接报错，阻止生成错误的 Write_Reg 计划。
* 已完成输出落盘：生成 `install/execplan.txt`（128bit 每行）与 `instructions_explained.txt`。
* 已完成 `instructions_explained.txt` 字段级增强：输出 `field_value_original_*` 与 `field_value_write_*`（bin/hex），用于直接对照字段值变化。
* 已完成安装清单输出规则更新：
    * 输出目录包含 `install/` 与 `install/cfg_pkg/`。
    * `sca_cfg.json` 的 `config/configs` 路径与 `cfg_pkg` 内复制后的 bitstream 文件一致。
    * 当存在 SFU 配置时，`sca_cfg.json` 额外输出 `opX_sfu_config`，并指向 `install/cfg_pkg/<SFU_TYPE>.txt`。
    * matrix 路径使用 `slice00` 这类两位编号格式。
    * 不再输出 `matrixB'`。
* 已完成可选 `Bank_data` 导出：
    * 通过 `-b` / `--export-bank-data` 启用。
    * 输出目录为 `<output_prefix>/Bank_data/`。
    * 输出文件命名为 `sliceXX_BankXX_data.txt`。
    * 每行为 32bit 大端 hex（`0xXXXXXXXX`），地址基于 bank 内 `base_addr=0`。
    * 依据地址规划规则按 128bit 粒度放置数据，空洞自动补 0。
    * 整个 bank 无数据时不输出该 bank 文件。
* 已完成 `max` 示例输入更新：当前示例按“仅输入 A，输出 D”组织。

### 待完成工作
* 控制寄存器自动计算函数仍为占位实现。当前框架已支持“根据算子形状计算控制寄存器值后再下发”，但实际每类算子的控制寄存器计算脚本尚未补齐。 已完成√
* 更多算子模板信息仍需补充到统一基础信息文件中，目前主要围绕示例算子完成联调。
* 尚未补充系统化测试：目前主要依赖样例 JSON 和真实 bitstream 做功能验证，缺少单元测试与回归测试。
* 尚未提供对“未启用寄存器但允许新写入”的可配置策略。当前行为是严格报错，适合先做错误计划拦截，但后续如果硬件侧允许某些寄存器冷启动写入，需要再扩展策略开关。



### 已知约束或问题
* 当模板中没有原始配置码流时，脚本无法判断寄存器是否来自已启用 chunk，此时不会执行“未启用寄存器”校验。
* 当前 README 中后续章节仍保留了较多设计说明和示例表格，其中有些内容是设计目标，不完全等同于当前实现状态；如需严格对齐代码行为，请优先参考本节“当前实现状态”。

### 运行示例

```bash
python main.py examples/sample_gemm_local_execution_input.json
```

启用 bank 数据导出：

```bash
python main.py examples/rmsnorm.json -b
```

等价长参数：

```bash
python main.py examples/rmsnorm.json --export-bank-data
```

可选输出归一化后的 JSON：

```bash
python main.py examples/sample_execution_input.json --dump-normalized-json normalized.json
```


## 输入json文件


### 当前 JSON 输入建议格式



```json
{
  "used_slices": 28,
  "operators": [
    {
      "id": "op0",
      "type": "prefill_gemm_local",
      "used_slices": "0b1111111111111111111111111111",
      "inputs": {
        "A": {
          "shape": [128, 128, 32],
          "remapping": null,
          "source": {
            "type": "external"
          }
        },
        "B": {
          "shape": [128, 128, 32],
          "remapping": null,
          "source": {
            "type": "external"
          }
        },
        "B'": {
          "shape": [128, 128, 32],
          "remapping": null,
          "source": {
            "type": "external"
          }
        }
      },
      "output": {
        "shape": [1, 128, 128],
        "remapping": null
      }
    },    
    {
      "id": "op1",
      "type": "prefill_max_fp32MN_fp32MN",
      "used_slices": "0b1111111111111111111111111111",
      "inputs": {
        "A": {
          "shape": [1, 128, 128],
          "remapping": null,
          "source": "op0"
        }
      },
      "output": {
        "shape": [1, 1, 128],
        "remapping": null
      }
    }
  ]
}

```

## 更新记录（新增功能与改动汇总）

以下内容用于集中记录本项目迭代过程中的关键能力与规则变更（截至 2026-04-14）。

### 一、地址规划与数据放置
* 新增自动地址规划：输入/输出 tensor 按“只增不回收”方式顺序分配地址。
* 支持按 `used_slices` 掩码进行每算子独立 slice 规划，未配置时继承顶层默认掩码。
* 地址规划按 16Byte 对齐执行，但 `sca_cfg.json` 中显示为真实 byte 地址（不再右移 4bit）。
* 新增 config 统一地址规划能力：
    * config 数据纳入地址规划流程，不再依赖模板中的静态地址字段。
    * config 放置在所有数据 tensor 之后。
    * `config_length` 的单位为 64bit 行数，config 起始地址强制对齐到 BANK Row 起点（`col=0, subword=0`）。
    * `config_length=0` 时不分配 config 地址，`Load_Config` 地址字段置 0。
    * 当 `config_length>0` 但缺少规划地址时，生成流程会显式报错，避免静默回退。
    * 每个 config 包按整行预留空间，确保下一包也从 row 起点开始。

### 二、B / B' 相关规则
* 新增 `B'` 输入支持（包括解析、地址映射、寄存器写入链路）。
* 地址规划中 `B` 与 `B'` 共用同一地址空间（同算子内 `B'` 复用 `B` 的分配结果）。
* `sca_cfg.json` 中不再单独输出 `matrixB'` 条目。

### 三、控制寄存器与映射能力
* 新增 tensor `remapping` 字段支持（A/B/B'/C/D）：
    * 允许 `null` 或长度 26 的重映射数组（元素范围 0..25）。
    * `remapping != null` 时自动写入 `stream_engine.stream.address_remapping`。
    * 按 26 x 5bit 进行打包（共 130bit）。
* 新增实例映射策略增强：
    * 支持通过 `config/<op_type>/mapping_review.json` 将逻辑实例名映射到物理实例名。
    * 对 `wr_stream` 与 `wr_stream0` 采用固定归一策略，统一映射为 `wr_stream0`。
* 修复并增强常量字段解析：
    * `iga_pe.inport*.cfg_constant_pos` 与 `ga_pe.inport*.constant` 使用 `const0/1/2` 地址映射规则。
    * 常量字段与普通顺序字段拆分处理，避免位段覆盖与误映射。
    * `original_value_bin` 的解释来源改为“字段原值投影”，提升可读性与调试准确性。

### 四、寄存器解析与校验
* 模块实例映射改为 `(大组名称, 模块名称)` 联合映射，解决同名模块歧义。
* 字段定位改为基于位宽（`Xbit`）顺序累加，不再依赖表1中的 `[high:low]` 文本。
* 新增/强化校验能力：
    * 指令 chunk 行数严格校验。
    * 位宽范围与字段值合法性校验。
    * 对未启用寄存器写入、地址越界等场景增加显式错误提示。

### 五、输出与交付格式
* 指令二进制输出文件名统一为 `execplan.txt`。
* 指令输出采用 128bit 一行（由两条 64bit 指令拼接）。
* 产物目录升级为 `install/`：
    * 指令输出路径为 `install/execplan.txt`。
    * 新增 `install/cfg_pkg/`，自动复制本次使用到的算子 bitstream 包。
    * `sca_cfg.json` 中 config 路径对齐 `cfg_pkg`，matrix slice 路径采用 `slice00` 两位编号。
* `instructions_explained.txt` 新增字段级原值/写值输出（bin/hex），便于和寄存器字段一一对照。
* 单算子场景兼容输出 `config`，多算子场景输出 `configs` 字段。

### 六、流程编排与模板联动
* pipeline 调整为先生成/读取算子模板，再执行地址规划，保证地址规划可直接使用模板参数（如 `config_length`）。
* `Load_Config` 指令中的地址字段改为来自地址规划结果，不再回退模板静态地址。

### 七、SFU 配置类型化与双 Load_Config
* `config_sfu` 从布尔语义升级为“SFU 类型”语义，支持：`GELU`、`REC`、`REC_SQRT`、`ReLU`、`Sigmoid`、`SiLU`、`SQRT`、`Tanh`、`Ex`（兼容 `Sigmiod` 别名）。
* 当 `config_sfu` 非空时，单算子自动生成两条 `Load_Config`：
    * 第一条：算子配置包（`Config_SFU=0`）。
    * 第二条：SFU 系数包（`Config_SFU=1`）。
* SFU 系数来源为 `config/SFU_Coeff/<type>.txt`，长度自动检测并参与地址规划。
* 交付产物中新增 `opX_sfu_config` manifest 条目，并将 SFU 文件复制到 `install/cfg_pkg/`。

### 八、Bank_data 导出
* 新增 CLI 开关：`-b` / `--export-bank-data`。
* 启用后从 `sca_cfg.json` 读取矩阵数据项并导出到 `<output_prefix>/Bank_data/`。
* 导出文件按 `(slice, bank)` 组织，命名为 `sliceXX_BankXX_data.txt`。
* 输出格式为每行 32bit 大端 hex；bank 内地址从 0 开始连续展开。
* 按 128bit 地址分配粒度填充空洞为 0；无数据 bank 不生成文件。
