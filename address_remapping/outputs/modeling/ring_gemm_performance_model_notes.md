# ring_gemm 性能分析方法图 Deck

- 运行样例：`examples/graphs/ring_gemm/ring_gemm_bias.json`
- 语言风格：中文叙事 + 英文变量/事件名
- 图形风格：抽象示意为主，不按真实 cycle 等比例绘制
- 首页性能表口径：latency=10296, hardware_measured=9005, simulator_error=14.34%
- 关键数值：A events=128，B events=512，B requests=4096，latency=10296
- 关键语义：A/B load-event ready 采用 full-group completion。
