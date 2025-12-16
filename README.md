Twitch 实时字幕系统 / Twitch Live Subtitles
Twitch 直播实时字幕工具（Real-time subtitles for Twitch streams）  
Pipeline: HLS → VAD → Whisper ASR → EN→ZH → Gradio UI

环境 / Requirements
- Python 3.10+
- ffmpeg (in PATH)
- yt-dlp (Cookie mode only)
- 建议 NVIDIA GPU + CUDA（显著降低延迟）；CPU 可运行但延迟更高
安装 / Install
- pip install -r requirements.txt
使用 / Usage
- 1.Cookie 模式：Twitch 用户名 + auth-token + persistent
- 2.Manual HLS：粘贴 .m3u8 链接

Non-Commercial License
Educational use only. Commercial use is prohibited.
