#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-api-python-client>=2.0.0",
# ]
# ///

"""
Test script to validate YouTube API key and test channel resolution.

Usage:
    ./youtube-titles/test_ytapi.py
"""

import os
import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def test_api_key():
    """Test if YouTube API key is working."""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("❌ YOUTUBE_API_KEY environment variable not found")
        return False
    
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        
        # Simple test - get channel info for a well-known channel
        response = yt.channels().list(
            part="snippet",
            forHandle="@FPLRaptor"
        ).execute()
        
        if response.get("items"):
            channel = response["items"][0]
            print(f"✅ API key working! Found channel: {channel['snippet']['title']}")
            print(f"   Channel ID: {channel['id']}")
            return True
        else:
            print("❌ API key works but couldn't find test channel")
            return False
            
    except HttpError as e:
        if e.resp.status == 403:
            print("❌ API key invalid or quota exceeded")
        else:
            print(f"❌ API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_channel_resolution():
    """Test resolving different channel URL formats."""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return False
        
    yt = build("youtube", "v3", developerKey=api_key)
    
    # Test channels from our config
    test_channels = [
        ("Handle URL", "https://www.youtube.com/@FPLRaptor", "FPL Raptor"),
        ("Channel ID", "https://www.youtube.com/channel/UCweDAlFm2LnVcOqaFU4_AGA", "FPL Mate"),
        ("Custom URL", "https://www.youtube.com/fplfocal", "FPL Focal"),
    ]
    
    print("\n🔍 Testing channel resolution:")
    print("=" * 50)
    
    for url_type, url, expected_name in test_channels:
        try:
            channel_id = None
            
            if "@" in url:
                # Handle URL - use forHandle
                handle = url.split("@")[1]
                response = yt.channels().list(
                    part="snippet",
                    forHandle=f"@{handle}"
                ).execute()
                
            elif "channel/" in url:
                # Channel ID URL - extract ID directly
                channel_id = url.split("channel/")[1]
                response = yt.channels().list(
                    part="snippet", 
                    id=channel_id
                ).execute()
                
            else:
                # Custom URL - use search
                search_query = url.split("/")[-1] or expected_name
                response = yt.search().list(
                    part="snippet",
                    type="channel",
                    q=search_query,
                    maxResults=1
                ).execute()
                
                if response.get("items"):
                    channel_id = response["items"][0]["snippet"]["channelId"]
                    # Get full channel info
                    response = yt.channels().list(
                        part="snippet",
                        id=channel_id
                    ).execute()
            
            if response.get("items"):
                channel = response["items"][0]
                resolved_id = channel_id or channel["id"]
                print(f"✅ {url_type}: {channel['snippet']['title']}")
                print(f"   URL: {url}")
                print(f"   ID: {resolved_id}")
                
                # Test getting recent videos
                uploads_response = yt.channels().list(
                    part="contentDetails",
                    id=resolved_id
                ).execute()
                
                if uploads_response.get("items"):
                    uploads_id = uploads_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
                    
                    videos_response = yt.playlistItems().list(
                        part="snippet,contentDetails",
                        playlistId=uploads_id,
                        maxResults=3
                    ).execute()
                    
                    video_count = len(videos_response.get("items", []))
                    print(f"   Recent videos: {video_count} found")
                    
                    if video_count > 0:
                        latest = videos_response["items"][0]
                        print(f"   Latest: {latest['snippet']['title'][:50]}...")
                        
            else:
                print(f"❌ {url_type}: Could not resolve {url}")
                
        except HttpError as e:
            print(f"❌ {url_type}: API error {e.resp.status}")
        except Exception as e:
            print(f"❌ {url_type}: Error - {e}")
            
        print()


def main():
    print("🧪 YouTube API Test Script")
    print("=" * 30)
    
    if not test_api_key():
        return 1
        
    test_channel_resolution()
    
    print("✅ All tests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())