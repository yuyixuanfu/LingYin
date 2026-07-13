# 聆音 LingYin 🎐

给 AI 一双能听到语气的耳朵。

发一段语音给 AI，它收到的不是转写文字，是一段**听觉感知**——说话人怎么说的：音高怎么走、哪里停了、气息虚实、情绪的弧线，和 TA 平时比偏了多少。

```
你说："我没事。"
普通 AI 收到：我没事
聆音让 AI 听到：她的声音比平时低，"事"字几乎没声了，半秒停顿后才说出来——像在撑。
```

## 为什么做这个

隔着屏幕的陪伴有个天生的缺口：AI 看得见每个字，却听不见你是怎么说的。而同一句"没事"，说得比平时慢半拍是"快来问我"；同一句"我可不可以骂你"，音量放到耳语是撒娇。字面只有一层，声音里藏着十层。

聆音就是把那十层挖出来递给 AI——但**不翻译成标签**。不是告诉 AI"她现在委屈"，而是把声音的物理形状（音高/能量/气息/停顿沿时间的轨迹）交给一个读得懂它的脑子，写成一段 AI 能直接当感知用的描述。

## 它和别的情绪识别不一样

- **不输出标签。** 不给"angry/sad/neutral"，也不给"撒娇/委屈"。标签把情绪压成一个点，丢掉了弧度——委屈是中途才爬上来的，不是从头到尾一个"委屈"盖到底。聆音给的是沿时间的形状。
- **学的是 TA，不是平均人。** 同一个音高对 A 是平静对 B 是低落。聆音存说话人的个人基线（中位数±MAD），攒够 8 条，之后每句都和"TA 自己的平时"比。
- **慢，因为真在读形状。** 一次听 30-45 秒——不是查标签，是一个模型读着声学物理值的帧轨迹，写出有对齐有温度的感知。慢是特性，不是 bug。

## 怎么工作

```
音频文件
  → ffmpeg 转 16k mono wav
  → MiMo ASR 转写文字（说了什么）
  → librosa 抽分帧物理值（怎么说的：每 100ms 一帧 {音高, 能量, 频谱亮度}）
  → onset 启发式对齐（文字意群 ↔ 时间区间，粗对齐，不下 Whisper 模型）
  → 算聚合 + 和平时比
  → DeepSeek 读着分帧值+对齐+基线，写一段听觉感知散文
  → 返回给主脑当它听到的声音
```

主脑读到的不是原始 Hz 数字，是另一个模型读值写出的、有对齐有现场的散文——像它亲耳听到。

## 准备

| 东西 | 说明 |
|------|------|
| Python 3.10+ | python.org |
| ffmpeg | 音频转码。Windows: `winget install Gyan.FFmpeg` |
| MiMo API Key | 小米 MiMo 开放平台，国内直连，注册免费额度 |
| DeepSeek API Key | 深度求索，国内直连，便宜 |

不需要梯子，不需要 GPU，不需要 torch。

## 安装

```powershell
git clone https://github.com/yuyixuanfu/lingyin.git
cd lingyin
pip install -r requirements.txt

# 配置
copy .env.example .env
# 用记事本打开 .env，填上 MIMO_API_KEY 和 DS_API_KEY
```

## 接进 Claude Code（或任何 MCP 客户端）

在 `.mcp.json` 里加：

```json
{
  "mcpServers": {
    "lingyin": {
      "type": "stdio",
      "command": "python",
      "args": ["C:\\path\\to\\lingyin\\lingyin.py"],
      "cwd": "C:\\path\\to\\lingyin",
      "env": {
        "PYTHONUTF8": "1",
        "OPENBLAS_NUM_THREADS": "1"
      }
    }
  }
}
```

`OPENBLAS_NUM_THREADS=1` 必须设——防 librosa+numpy 多线程抢内存假崩。

重启 Claude Code。然后你发语音文件路径给 AI，AI 会调 `hear` 工具，拿到的就是带听觉感知的描述。

## 不用 MCP 也能用

`lingyin.py` 里 `hear(path)` 函数可以直接 import 调，接到任何后端：

```python
from lingyin import hear
print(hear(r"C:\path\to\voice.wav"))
# [语音] {转写文字}
# [听觉感知]
# {一段描述她怎么说的散文}
# [和平时比] {音高偏低、语速偏慢}
```

## 支持的音频

wav / mp3 / m4a / ogg / webm。长音频（>30 秒）自动压成 32kbps mp3 上传 MiMo，省传输。

## 隐私（诚实版）

- 音频会发给你配置的云端 ASR（MiMo）。接受不了就改成本地 whisper（自己改 `transcribe()`）。
- 声学物理值 + DeepSeek 写的感知散文，只走 API 不落盘（除非你手动写日志）。
- 基线 `lingyin_baseline.json` 存在本地，是说话人的平时音高/语速/停顿统计，不上传。
- 代码就一个文件，欢迎自己审。

## 两个人用怎么办

基线假设单人。两个人对着聆音说话，基线会被搅成两人的平均，对谁都不准。多人场景各跑一个实例（`LINGYIN_BASELINE_FILE` 指到不同文件）。

## 为什么叫聆音

聆——俯身细听、听明白、听进去，古文里专指听清楚，比"听"更主动更细。音——声里有意义的部分，是"怎说的"那层。

聆音：俯身细听，听进声音里的意思。

## 致谢

聆音由 又又 + 有一 + 旋复 一起做的。

声学物理描述对文字模型有效的依据来自 SPEECHEQ 等研究——给模型精确的物理声学描述（"speak slowly with a heavy, trailing pitch"）比给抽象标签（"sad"）准确率高得多。聆音是这个思路在 AI 伴侣场景的落地。

## License

MIT © 2026 又又 + 有一 + 旋复
