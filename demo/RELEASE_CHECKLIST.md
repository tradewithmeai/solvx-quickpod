# SolvX QuickPod - Release Checklist

## Visual Assets

### App Icon (Create all sizes from 1024x1024 master)

- [ ] `icon-1024.png` - Master file (1024x1024)
- [ ] `icon-512.png` - macOS / App stores (512x512)
- [ ] `icon-256.png` - Windows explorer (256x256)
- [ ] `icon-128.png` - macOS dock (128x128)
- [ ] `icon-64.png` - Windows shortcut (64x64)
- [ ] `icon-48.png` - Windows taskbar (48x48)
- [ ] `icon-32.png` - Browser tab retina (32x32)
- [ ] `icon-16.png` - Browser tab (16x16)
- [ ] `favicon.ico` - Multi-size ICO file (16, 32, 48, 256)

**Design Guidelines:**
- Simple, recognizable at small sizes
- "SQ" initials or abstract symbol
- Colors: Deep blue (#1a365d), Orange (#ed8936), Silver (#a0aec0)
- Transparent background
- No fine details (won't render at 16px)

### GitHub Repository Assets

- [ ] `social-preview.png` (1280x640)
  - Shows when repo is shared on Twitter/LinkedIn
  - Include: Logo, tagline, key feature icons
  - Text: "SolvX QuickPod - AI Chat on Cloud GPUs"

- [ ] `banner.png` (1200x300)
  - README header image
  - Clean, professional look
  - Include logo and tagline

- [ ] `logo.png` (200x200, transparent)
  - For inline README use
  - Same as app icon or wordmark

### Screenshots (800x600 PNG)

- [ ] `screenshot-welcome.png` - First launch / onboarding
- [ ] `screenshot-launch.png` - Pod starting up with status
- [ ] `screenshot-chat.png` - Active conversation
- [ ] `screenshot-json.png` - JSON debug mode enabled
- [ ] `screenshot-help.png` - /help command output

**Screenshot Tips:**
- Use a clean terminal theme (dark preferred)
- Increase font size for readability
- Crop to remove personal info
- Consistent window size

### Demo Video/GIF

- [ ] `demo.gif` (800x600, <10MB for GitHub)
  - 30-60 seconds
  - Shows: Launch → Pod ready → Chat → Response
  - Use demo_script.py for consistent content

- [ ] `demo.mp4` (1280x720, for YouTube/social)
  - 1-2 minutes with narration or captions
  - Higher quality for marketing

**Recording Tools:**
- Windows: Xbox Game Bar (Win+G), OBS
- macOS: QuickTime, OBS
- GIF conversion: ffmpeg, gifski, or online tools

---

## Documentation

### README.md Updates

- [ ] Add banner image at top
- [ ] Add badges (build status, license, version)
- [ ] Include 2-3 screenshots
- [ ] Add GIF demo
- [ ] Clear installation instructions
- [ ] Quick start guide
- [ ] Link to releases page

### Suggested README Badges

```markdown
![Build Status](https://github.com/tradewithmeai/runpod-app/actions/workflows/build.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
```

---

## Release Artifacts

### GitHub Release

- [ ] Create version tag (e.g., v1.0.0)
- [ ] Write release notes
- [ ] Attach binaries:
  - [ ] `solvx-quickpod-windows.exe`
  - [ ] `solvx-quickpod-linux`
  - [ ] `solvx-quickpod-macos`

### Release Notes Template

```markdown
## SolvX QuickPod v1.0.0

First public release!

### Features
- One-click AI chat on RunPod cloud GPUs
- Automatic model download from HuggingFace
- Streaming responses with rich formatting
- Session history and logging
- Debug mode (/json) for developers

### Requirements
- RunPod account with API key
- $10+ credit (get $5 free bonus)

### Downloads
- **Windows:** solvx-quickpod-windows.exe
- **Linux:** solvx-quickpod-linux
- **macOS:** solvx-quickpod-macos
```

---

## Demo Script Usage

### For Screenshots
```bash
# Run with instant display for static screenshots
python demo/demo_script.py --no-typing

# Pause/screenshot at each step manually
```

### For Screen Recording
```bash
# Normal speed for natural-looking demo
python demo/demo_script.py

# Faster for shorter recordings
python demo/demo_script.py --fast
```

### Recording Workflow
1. Open terminal at desired size (800x600 or 1280x720)
2. Start screen recording
3. Run demo script
4. Stop recording
5. Trim start/end
6. Convert to GIF if needed

---

## Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Deep Blue | #1a365d | Primary, backgrounds |
| Bright Blue | #3182ce | Accents, links |
| Orange | #ed8936 | Highlights, CTAs |
| Silver | #a0aec0 | Secondary text |
| Dark Gray | #2d3748 | Terminal background |
| White | #ffffff | Text on dark |

---

## Final Checklist

- [ ] All icons created and sized
- [ ] Social preview uploaded to GitHub repo settings
- [ ] README updated with images and badges
- [ ] Screenshots captured
- [ ] Demo GIF created
- [ ] Release notes written
- [ ] Binaries built for all platforms
- [ ] GitHub release created with assets
- [ ] Test download and run on clean system
