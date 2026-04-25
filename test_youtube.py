#!/usr/bin/env python3
"""
Diagnostic script to test YouTube stream URL resolution
"""
import sys
import importlib
import cv2

# Test YouTube URLs
YOUTUBE_URLS = {
    "fresno": "https://www.youtube.com/watch?v=1xl0hX-nF2E",
    "miami": "https://www.youtube.com/watch?v=gCKAn2q35Dw",
    "jackson_hole": "https://www.youtube.com/live/1EiC9bvVGnk?si=fcE4ToeMHOD-ZRS7"
}

def get_youtube_direct_url(youtube_url):
    """Attempt to resolve YouTube URL using yt_dlp or youtube_dl"""
    print(f"\n🔍 Testing: {youtube_url}")
    try:
        yt_dlp = importlib.import_module("yt_dlp")
        YoutubeDL = getattr(yt_dlp, "YoutubeDL")
        print("   Using: yt_dlp ✓")
    except (ImportError, ModuleNotFoundError):
        try:
            youtube_dl = importlib.import_module("youtube_dl")
            YoutubeDL = getattr(youtube_dl, "YoutubeDL")
            print("   Using: youtube_dl (fallback)")
        except (ImportError, ModuleNotFoundError):
            print("   Neither yt_dlp nor youtube_dl installed!")
            print("   Install with: pip install yt-dlp")
            return None

    ydl_opts = {
        "format": "best",
        "quiet": False,
        "skip_download": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,
    }

    try:
        print("   Extracting stream info...")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if info is None:
                print("   ❌ Info is None")
                return None

            if "formats" in info:
                print(f"   Found {len(info['formats'])} formats")
                formats = sorted(
                    (f for f in info["formats"] if f.get("url")),
                    key=lambda item: item.get("height", 0) or 0,
                    reverse=True,
                )
                for idx, fmt in enumerate(formats[:3]):
                    height = fmt.get("height", "?")
                    print(f"     Format {idx}: height={height}")
                
                for fmt in formats:
                    url = fmt.get("url")
                    if url:
                        print(f"   ✅ Resolved to direct URL")
                        return url

            resolved_url = info.get("url")
            if resolved_url:
                print(f"   ✅ Got URL from info")
                return resolved_url
            else:
                print(f"   ❌ No URL found in info")
                return None
    except Exception as exc:
        print(f"   ❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        return None

def test_cv2_open(url):
    """Test if cv2.VideoCapture can open the URL"""
    print(f"   Testing cv2.VideoCapture...")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"   ✅ cv2.VideoCapture opened successfully!")
        cap.release()
        return True
    else:
        print(f"   ❌ cv2.VideoCapture failed to open")
        return False

print("=" * 80)
print("YouTube Stream URL Diagnostic Test")
print("=" * 80)

for name, url in YOUTUBE_URLS.items():
    print(f"\n📺 Testing: {name}")
    direct_url = get_youtube_direct_url(url)
    if direct_url:
        # Optionally test cv2 (might be slow)
        print(f"   Skipping cv2 test (might be slow for live streams)")
    else:
        print(f"   ❌ Could not resolve URL for {name}")

print("\n" + "=" * 80)
print("Diagnostic Summary:")
print("=" * 80)
print("If URLs failed to resolve:")
print("  1. Check internet connection")
print("  2. Ensure yt_dlp is installed: pip install yt-dlp")
print("  3. Check if YouTube streams are still live")
print("  4. Try updating yt_dlp: pip install --upgrade yt-dlp")
print("\nFor live stream processing, ensure:")
print("  - yt_dlp or youtube_dl is installed")
print("  - Internet connection is stable")
print("  - YouTube streams are currently live")
print("=" * 80)
