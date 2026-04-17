import requests
import pandas as pd
import random

def scrape_tiktok_url(url, num_comments=50):
    """
    Ekstrak data nyata dari TikTok menggunakan free open API TikWM.
    Catatan: Mengubah sistem Random Mock menjadi Real Scraper.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
    }
    
    # 1. Fetch Video Stats
    try:
        vid_res = requests.post("https://tikwm.com/api/", data={"url": url}, headers=headers, timeout=10)
        vid_data = vid_res.json()
        
        if vid_data.get("code") != 0 or "data" not in vid_data:
            # Fallback ke nilai statis jika API limit/error
            raise Exception("Gagal menarik metrik video dari API (Rate limit atau URL tidak valid)")
            
        video_info = vid_data["data"]
        
        video_likes = video_info.get("digg_count", 1500)
        video_comments_count = video_info.get("comment_count", 200)
        video_shares = video_info.get("share_count", 50)
        
        # Duration info from TikWM is in 'duration' (seconds). If not found, default to 15
        video_watch_duration = float(video_info.get("duration", 15.0))
        # Since 'duration' is the full video length, we estimate 'watch duration' as roughly 70% of it, 
        # or just use the length directly as requested feature.
        if video_watch_duration <= 0:
            video_watch_duration = 15.0
            
    except Exception as e:
        print(f"Error Stat Fetch: {e}")
        # Default jika gagal
        video_likes = 5000
        video_comments_count = 100
        video_shares = 10
        video_watch_duration = 30.0

    # 2. Fetch Real Comments
    scraped_comments = []
    try:
        comm_res = requests.get(f"https://tikwm.com/api/comment/list/?url={url}&count={num_comments}", headers=headers, timeout=10)
        comm_data = comm_res.json()
        
        if comm_data.get("code") == 0 and "data" in comm_data and "comments" in comm_data["data"]:
            for c in comm_data["data"]["comments"]:
                text = c.get("text", "")
                username = c.get("user", {}).get("nickname", "anonim")
                
                scraped_comments.append({
                    "username": username,
                    "text": text,
                    "likes": video_likes, 
                    "comments": video_comments_count,
                    "shares": video_shares,
                    "watch_duration": video_watch_duration
                })
        else:
             raise Exception("Komentar kosong atau gagal ditarik.")
    except Exception as e:
        print(f"Error Comment Fetch: {e}")
        # Jika gagal atau komentar dinonaktifkan di video, fallback ke pesan error pada dataframe
        scraped_comments.append({
            "username": "System",
            "text": "Tidak dapat menarik komentar (API Rate Limit / Komentar Dinonaktifkan).",
            "likes": video_likes,
            "comments": video_comments_count,
            "shares": video_shares,
            "watch_duration": video_watch_duration
        })

    df_scraped = pd.DataFrame(scraped_comments)
    return {
        "url": url,
        "video_metrics": {
            "likes": video_likes,
            "comments_count": video_comments_count,
            "shares": video_shares,
            "watch_duration": video_watch_duration
        },
        "comments_df": df_scraped
    }
