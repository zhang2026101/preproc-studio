"""启动入口

    python run.py            # 默认 127.0.0.1:8770
    python run.py --port 9000
    python run.py --host 0.0.0.0     # 局域网给同事看
"""

import argparse
import os
import sys

import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    print(f"\n  Preproc Studio  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
