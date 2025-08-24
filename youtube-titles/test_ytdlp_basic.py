#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "yt-dlp>=2024.8.0",
#   "python-dateutil>=2.9.0.post0",
# ]
# ///

"""
Basic yt-dlp functionality test for FPL Video Picker migration.
Tests core functionality before full migration.
"""

from datetime import UTC, datetime

import yt_dlp


def test_single_video_metadata():
    """Test extracting metadata from a single YouTube video."""
    print("=== Testing Single Video Metadata Extraction ===")

    # Use a well-known stable video for testing
    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Extracting info from: {test_video_url}")
            info = ydl.extract_info(test_video_url, download=False)

            # Extract key metadata we need for FPL video picker
            metadata = {
                "title": info.get("title", "N/A"),
                "uploader": info.get("uploader", "N/A"),  # This is channel name
                "upload_date": info.get("upload_date", "N/A"),
                "description": info.get("description", "")[:200] + "..."
                if info.get("description")
                else "N/A",
                "webpage_url": info.get("webpage_url", "N/A"),
                "view_count": info.get("view_count", "N/A"),
                "duration": info.get("duration", "N/A"),
            }

            print("✅ Single video metadata extraction successful!")
            print(f"Title: {metadata['title']}")
            print(f"Channel: {metadata['uploader']}")
            print(f"Upload Date: {metadata['upload_date']}")
            print(f"URL: {metadata['webpage_url']}")
            print(f"Description: {metadata['description'][:100]}...")

            # Test date parsing
            if metadata["upload_date"] != "N/A":
                parsed_date = datetime.strptime(
                    metadata["upload_date"], "%Y%m%d"
                ).replace(tzinfo=UTC)
                print(f"Parsed Date: {parsed_date}")
                print("✅ Date parsing successful!")

            return True

    except Exception as e:
        print(f"❌ Single video test failed: {e}")
        return False


def test_channel_video_listing(channel_url: str, max_videos: int = 3) -> dict | None:
    """Test listing recent videos from a YouTube channel."""
    print(f"\n=== Testing Channel Video Listing: {channel_url} ===")

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Faster for listing
            "playlistend": max_videos,  # Limit number of videos
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Extracting channel info from: {channel_url}")
            info = ydl.extract_info(channel_url, download=False)

            channel_name = info.get("uploader", info.get("channel", "Unknown"))
            entries = info.get("entries", [])

            print("✅ Channel extraction successful!")
            print(f"Channel Name: {channel_name}")
            print(f"Videos Found: {len(entries)}")

            # Test getting detailed info for first video
            if entries:
                first_video = entries[0]
                print(f"\nFirst Video: {first_video.get('title', 'No Title')}")
                print(f"Video URL: {first_video.get('url', 'No URL')}")
                print(f"Upload Date: {first_video.get('upload_date', 'No Date')}")

                # Try to get full metadata for first video
                if first_video.get("url"):
                    try:
                        detailed_opts = {"quiet": True, "no_warnings": True}
                        with yt_dlp.YoutubeDL(detailed_opts) as detail_ydl:
                            detailed_info = detail_ydl.extract_info(
                                first_video["url"], download=False
                            )
                            print(f"Full Title: {detailed_info.get('title', 'N/A')}")
                            print(
                                f"Description Length: {len(detailed_info.get('description', ''))}"
                            )
                            print("✅ Detailed video metadata extraction successful!")
                    except Exception as e:
                        print(f"⚠️  Detailed video extraction failed: {e}")

            return {
                "channel_name": channel_name,
                "video_count": len(entries),
                "videos": entries[:3],  # Return first 3 for inspection
            }

    except Exception as e:
        print(f"❌ Channel test failed: {e}")
        return None


def test_fpl_channels():
    """Test all FPL channels used in the video picker."""
    print("\n=== Testing FPL Channels ===")

    fpl_channels = [
        "https://www.youtube.com/@FPLRaptor",
        "https://www.youtube.com/channel/UCweDAlFm2LnVcOqaFU4_AGA",  # FPL Mate
        "https://www.youtube.com/channel/UCxeOc7eFxq37yW_Nc-69deA",  # FPL Andy
        "https://www.youtube.com/fplfocal",
        "https://www.youtube.com/channel/UCcPWnCj5AKC19HaySZjb25g",  # FP Harry
    ]

    results = {}
    successful_channels = 0

    for channel_url in fpl_channels:
        result = test_channel_video_listing(channel_url, max_videos=2)
        if result:
            successful_channels += 1
            results[channel_url] = result
        else:
            results[channel_url] = None

    print("\n=== FPL Channels Test Summary ===")
    print(f"Successful: {successful_channels}/{len(fpl_channels)} channels")

    for channel_url, result in results.items():
        if result:
            print(f"✅ {result['channel_name']}: {result['video_count']} videos")
        else:
            print(f"❌ {channel_url}: Failed")

    return (
        successful_channels >= len(fpl_channels) // 2
    )  # Success if at least half work


def test_metadata_structure_compatibility():
    """Test that we can create the same VideoItem structure from yt-dlp data."""
    print("\n=== Testing Metadata Structure Compatibility ===")

    # Test with FPL channel but without extract_flat to get actual video URLs
    test_url = "https://www.youtube.com/@FPLRaptor"

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,  # Need full extraction to get video URLs
            "playlistend": 1,  # Just get one video for testing
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            channel_info = ydl.extract_info(test_url, download=False)

            if not channel_info.get("entries"):
                print("❌ No videos found for compatibility test")
                return False

            # Get first video entry (should have full info since extract_flat=False)
            first_video = channel_info["entries"][0]

            # Create VideoItem-compatible structure
            video_item_data = {
                "title": first_video.get("title", ""),
                "url": first_video.get("webpage_url", first_video.get("url", "")),
                "published_at": datetime.strptime(
                    first_video.get("upload_date", "20240101"), "%Y%m%d"
                ).replace(tzinfo=UTC),
                "channel_name": first_video.get(
                    "uploader", channel_info.get("uploader", "Unknown")
                ),
                "description": first_video.get("description", "") or "",
            }

            print("✅ VideoItem structure compatibility successful!")
            print("Compatible structure created:")
            print(f"  Title: {video_item_data['title'][:50]}...")
            print(f"  URL: {video_item_data['url']}")
            print(f"  Published: {video_item_data['published_at']}")
            print(f"  Channel: {video_item_data['channel_name']}")
            print(f"  Description Length: {len(video_item_data['description'])}")

            return True

    except Exception as e:
        print(f"❌ Metadata structure compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all yt-dlp functionality tests."""
    print("🚀 Starting yt-dlp Functionality Tests for FPL Video Picker Migration\n")

    tests_passed = 0
    total_tests = 4

    # Test 1: Single video metadata
    if test_single_video_metadata():
        tests_passed += 1

    # Test 2: Channel listing (with a reliable channel)
    test_channel_result = test_channel_video_listing(
        "https://www.youtube.com/@youtube", max_videos=2
    )  # Official YouTube channel
    if test_channel_result:
        tests_passed += 1

    # Test 3: FPL channels specifically
    if test_fpl_channels():
        tests_passed += 1

    # Test 4: Metadata structure compatibility
    if test_metadata_structure_compatibility():
        tests_passed += 1

    print(f"\n{'=' * 50}")
    print(f"🎯 TEST RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed >= 3:
        print("✅ MIGRATION GO/NO-GO: ✅ GO - yt-dlp is viable for migration")
        print("Recommendation: Proceed with full migration to yt-dlp")
        return 0
    elif tests_passed >= 2:
        print("⚠️  MIGRATION GO/NO-GO: ⚠️  CAUTION - Some issues detected")
        print("Recommendation: Investigate failures before full migration")
        return 1
    else:
        print("❌ MIGRATION GO/NO-GO: ❌ NO-GO - Too many failures")
        print("Recommendation: Do not migrate to yt-dlp at this time")
        return 2


if __name__ == "__main__":
    exit(main())
