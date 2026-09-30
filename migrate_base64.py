"""
把 products.json 里的 base64 图片/视频迁移到 uploads/ 目录，字段替换成 URL。

用法：
    python migrate_base64.py

特点：
- 自动备份原 products.json 到 backup/
- 支持主图、副图、视频、详情 HTML 内嵌图片
- 幂等：已经是 URL 的字段会跳过，重复运行不会重复迁移
- 迁移失败的资源保留原值（不丢数据）
"""

import json
import base64
import re
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================
DATA_FILE = Path("products.json")
UPLOAD_DIR = Path("uploads").resolve()
BACKUP_DIR = Path("backup")

# ★ 迁移后的 URL 前缀
# 本地开发：保持 http://localhost:8000
# Railway：改成你的线上地址，例如 https://ai-ecommerce-backend-production-0776.up.railway.app
BACKEND_BASE_URL = "https://ai-ecommerce-backend-production-0776.up.railway.app"

# data URL 正则：data:image/jpeg;base64,xxxxx 或 data:video/mp4;base64,xxxxx
DATA_URL_PATTERN = re.compile(
    r'^data:(image|video)/([\w.+-]+);base64,(.+)$',
    re.DOTALL
)

# MIME 子类型 → 文件扩展名
EXT_MAP = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "webp": "webp",
    "gif": "gif",
    "bmp": "bmp",
    "svg+xml": "svg",
    "mp4": "mp4",
    "webm": "webm",
    "quicktime": "mov",
    "x-msvideo": "avi",
    "ogg": "ogv",
}


# ============================================================
# 工具函数
# ============================================================
def safe_filename(name: str) -> str:
    """把商品名清洗成安全的文件名。"""
    return "".join(c for c in (name or "unknown") if c.isalnum() or c in "-_") or "unknown"


def save_data_url_to_file(data_url: str, category: str, product_name: str) -> str:
    """
    把 data URL 解码后存到 uploads/，返回完整公开 URL。

    - data_url 不是 data: 开头 → 原样返回（已经是 URL 或空）
    - 解码失败 → 原样返回（不丢数据）
    - 成功 → 返回 BACKEND_BASE_URL + /uploads/...
    """
    if not data_url or not isinstance(data_url, str):
        return data_url

    if not data_url.startswith("data:"):
        # 已经是 http(s):// 或相对路径，直接返回
        return data_url

    m = DATA_URL_PATTERN.match(data_url)
    if not m:
        print(f"    ⚠️ 无法解析的 data URL（前缀：{data_url[:60]}...）")
        return data_url

    mime_main, mime_sub, b64_data = m.groups()
    ext = EXT_MAP.get(mime_sub.lower(), mime_sub.lower())

    # 解码
    try:
        content = base64.b64decode(b64_data)
    except Exception as e:
        print(f"    ❌ base64 解码失败：{e}")
        return data_url

    # 决定落盘目录
    if category == "sub":
        target_dir = UPLOAD_DIR / "sub" / safe_filename(product_name)
    else:
        target_dir = UPLOAD_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)

    # 用时间戳 + 随机片段避免重名
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    filename = f"migrated_{stamp}.{ext}"
    target_path = target_dir / filename

    target_path.write_bytes(content)

    # 拼 URL
    if category == "sub":
        rel = f"sub/{safe_filename(product_name)}/{filename}"
    else:
        rel = f"{category}/{filename}"

    return f"{BACKEND_BASE_URL.rstrip('/')}/uploads/{rel}"


def migrate_detail_html(html: str, product_name: str) -> tuple[str, int]:
    """
    把详情 HTML 里的所有 data:image base64 替换成 URL。
    返回 (新 HTML, 迁移数量)。
    """
    if not html or "data:image" not in html:
        return html, 0

    count = [0]

    # 匹配 src="data:image/...;base64,xxx" 或 src='data:image/...;base64,xxx'
    pattern = re.compile(
        r'(["\'])data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+\1',
        re.DOTALL
    )

    def replace_src(match):
        full = match.group(0)
        quote = match.group(1)
        # 去掉两端的引号
        data_url = full[1:-1]
        new_url = save_data_url_to_file(data_url, "detail", product_name)
        if new_url != data_url:
            count[0] += 1
        return f"{quote}{new_url}{quote}"

    new_html = pattern.sub(replace_src, html)
    return new_html, count[0]


# ============================================================
# 主流程
# ============================================================
def migrate():
    print("=" * 60)
    print("🚀 开始迁移 base64 → URL")
    print("=" * 60)
    print(f"📂 数据文件：{DATA_FILE.resolve()}")
    print(f"📂 上传目录：{UPLOAD_DIR}")
    print(f"🌐 URL 前缀：{BACKEND_BASE_URL}")
    print()

    # ---------- 1. 检查 ----------
    if not DATA_FILE.exists():
        print(f"❌ 找不到 {DATA_FILE}，请确认脚本在 main.py 同级目录下运行")
        return

    # ---------- 2. 确保上传目录存在 ----------
    for sub in ("main", "sub", "video", "detail"):
        (UPLOAD_DIR / sub).mkdir(parents=True, exist_ok=True)

    # ---------- 3. 备份 ----------
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"products_backup_{stamp}.json"
    shutil.copy(DATA_FILE, backup_path)
    original_size = DATA_FILE.stat().st_size
    print(f"📦 已备份到：{backup_path}")
    print(f"📏 原文件大小：{original_size / 1024 / 1024:.2f} MB")
    print()

    # ---------- 4. 读取 ----------
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"📦 共 {len(products)} 个商品待处理\n")

    # ---------- 5. 逐个处理 ----------
    stats = {
        "main": 0,
        "sub": 0,
        "video": 0,
        "detail": 0,
        "skipped": 0,
        "failed": 0,
    }

    for p in products:
        name = p.get("name", "unknown")
        changes = []

        # 主图
        main = p.get("main_image", "")
        if isinstance(main, str) and main.startswith("data:"):
            new_url = save_data_url_to_file(main, "main", name)
            if new_url != main:
                p["main_image"] = new_url
                stats["main"] += 1
                changes.append("主图")
            else:
                stats["failed"] += 1
        elif main:
            stats["skipped"] += 1

        # 副图
        subs = p.get("sub_images", [])
        if isinstance(subs, list):
            new_subs = []
            sub_changed = 0
            for i, sub in enumerate(subs):
                if isinstance(sub, str) and sub.startswith("data:"):
                    new_url = save_data_url_to_file(sub, "sub", name)
                    if new_url != sub:
                        new_subs.append(new_url)
                        stats["sub"] += 1
                        sub_changed += 1
                    else:
                        new_subs.append(sub)
                        stats["failed"] += 1
                else:
                    new_subs.append(sub)
            if sub_changed:
                changes.append(f"副图×{sub_changed}")
            p["sub_images"] = new_subs

        # 视频
        video = p.get("video", "")
        if isinstance(video, str) and video.startswith("data:"):
            new_url = save_data_url_to_file(video, "video", name)
            if new_url != video:
                p["video"] = new_url
                stats["video"] += 1
                changes.append("视频")
            else:
                stats["failed"] += 1
        elif video:
            stats["skipped"] += 1

        # 详情 HTML（兼容 detail 和 detail_html 两个字段）
        for field in ("detail_html", "detail"):
            detail = p.get(field, "")
            if isinstance(detail, str) and "data:image" in detail:
                new_detail, cnt = migrate_detail_html(detail, name)
                if cnt > 0:
                    p[field] = new_detail
                    stats["detail"] += cnt
                    changes.append(f"详情图×{cnt}")

        if changes:
            print(f"  ✅ {name}：{', '.join(changes)}")

    # ---------- 6. 写回 ----------
    print()
    print("💾 正在写回 products.json ...")
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    new_size = DATA_FILE.stat().st_size

    # ---------- 7. 汇总 ----------
    print()
    print("=" * 60)
    print("✅ 迁移完成")
    print("=" * 60)
    print(f"主图      迁移：{stats['main']} 个")
    print(f"副图      迁移：{stats['sub']} 张")
    print(f"视频      迁移：{stats['video']} 个")
    print(f"详情图    迁移：{stats['detail']} 张")
    print(f"已是 URL  跳过：{stats['skipped']} 个")
    print(f"迁移失败       ：{stats['failed']} 个")
    print()
    print(f"📏 原文件大小：{original_size / 1024 / 1024:.2f} MB")
    print(f"📏 新文件大小：{new_size / 1024 / 1024:.2f} MB")
    if original_size > 0:
        saved = (1 - new_size / original_size) * 100
        print(f"📉 压缩率    ：{saved:.1f}%")
    print()
    print(f"📁 素材文件已存到：{UPLOAD_DIR}")
    print(f"💾 原文件已备份到：{backup_path}")
    print()
    print("⚠️ 提示：")
    print("  1. 确认无误后可删除 backup/ 目录")
    print("  2. 如果 URL 前缀写错了，可删掉 products.json 后从 backup 恢复，改完脚本再跑一次")
    print("  3. 迁移后的图片文件按 /uploads/{main|sub|video|detail}/ 分类存放")


if __name__ == "__main__":
    migrate()
