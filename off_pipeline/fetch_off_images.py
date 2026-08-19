# -*- coding: utf-8 -*-
"""
第 5 步：下载商品正面图（OFF 图片为 CC BY-SA，官方 CDN）。

  * 只下载全集内商品（items.jsonl），跳过已存在文件 -> 可断点续跑。
  * 4 线程 + 每请求 0.3s 间隔，对 CDN 保持礼貌；失败记录到 images_failed.txt，
    最终缺图由 item_image_mask 承接（这正是模型的模态补全模块要处理的场景）。

    python fetch_off_images.py
"""
import json
import queue
import threading
import time
import urllib.request

from off_common import IMG_DIR, PROC_DIR, UA

N_THREADS = 4
PAUSE = 0.3


def worker(q, failed, lock, stats):
    while True:
        try:
            idx, url = q.get_nowait()
        except queue.Empty:
            return
        out = IMG_DIR / f"{idx}.jpg"
        if out.exists() and out.stat().st_size > 0:
            with lock:
                stats["skip"] += 1
            q.task_done()
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 500:
                raise ValueError(f"too small ({len(data)}B)")
            out.write_bytes(data)
            with lock:
                stats["ok"] += 1
        except Exception as e:
            with lock:
                failed.append(f"{idx}\t{url}\t{type(e).__name__}: {e}")
                stats["fail"] += 1
        with lock:
            n = stats["ok"] + stats["fail"] + stats["skip"]
            if n % 500 == 0:
                print(f"  {n} 处理 | 成功 {stats['ok']} 失败 {stats['fail']} "
                      f"跳过 {stats['skip']}", flush=True)
        time.sleep(PAUSE)
        q.task_done()


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    n_no_url = 0
    for line in (PROC_DIR / "items.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("image_url"):
            q.put((r["idx"], r["image_url"]))
        else:
            n_no_url += 1
    total = q.qsize()
    print(f"待下载 {total} 张（无图片 URL: {n_no_url}）")

    failed, lock = [], threading.Lock()
    stats = {"ok": 0, "fail": 0, "skip": 0}
    threads = [threading.Thread(target=worker, args=(q, failed, lock, stats),
                                daemon=True) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (PROC_DIR / "images_failed.txt").write_text("\n".join(failed),
                                                encoding="utf-8")
    print(f"完成: 成功 {stats['ok']} | 失败 {stats['fail']} | "
          f"已存在跳过 {stats['skip']}")


if __name__ == "__main__":
    main()
