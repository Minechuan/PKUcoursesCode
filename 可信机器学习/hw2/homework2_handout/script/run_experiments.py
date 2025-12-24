#!/usr/bin/env python3
import subprocess, re, os
import numpy as np

def main():
    if not os.path.exists("record.txt"):
        open("record.txt","w").close()
    # alpha=0.7 
    x=0.7 #        for y in np.arange(0.2, 3.0, 0.2):

    for y in np.arange(0.0, 0.2, 0.05):

            cmd = ["python", "mc.py", f"--alp={x}", f"--lam={y}"]
            print(cmd)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("mc.py failed:", result.stderr)
                continue

            # 将完整输出写入 tmp.txt
            with open("tmp.txt", "w") as tmp_file:
                tmp_file.write(result.stdout)

            # 从 tmp.txt 中读取所有内容并按行拆分
            with open("tmp.txt", "r") as tmp_file:
                out = tmp_file.read().splitlines()



            natural_acc = None
            robust_acc = None

            # 从后向前查找匹配 tensor(数字)
            for line in reversed(out):
                if natural_acc is None and "natural_accuracy" in line:
                    m = re.search(r"tensor\(\s*([0-9]*\.?[0-9]+)", line)
                    if m: natural_acc = float(m.group(1))
                if robust_acc is None and "robustness" in line:
                    m = re.search(r"tensor\(\s*([0-9]*\.?[0-9]+)", line)
                    if m: robust_acc = float(m.group(1))
                if natural_acc is not None and robust_acc is not None:
                    break

            if natural_acc is None or robust_acc is None:
                print(f"[Warning] 未能解析指标: alp={x}, lam={y}")
                continue

            score = natural_acc / 2 + robust_acc

            with open("record.txt","a") as f:
                f.write(
                    f"alpha={x:.2f} lambda={y:.2f} "
                    f"natural_accuracy={natural_acc:.4f} "
                    f"robustness={robust_acc:.4f} "
                    f"score={score:.4f}\n"
                )

if __name__ == "__main__":
    main()
