#!/usr/bin/env python3
import subprocess, re, os
import numpy as np

def main():
    if not os.path.exists("record.txt"):
        open("record.txt","w").close()
     

    for y in [1,4,8,16,32]:

            cmd = ["python", "exp_lora.py", f"--r={16}", f"--a={y}"]
            print(cmd)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("exp_lora.py failed:", result.stderr)
                continue

            # 将完整输出写入 tmp.txt
            with open("output.txt", "a") as tmp_file:  # 使用 "a" 模式
                tmp_file.write(result.stdout)
                tmp_file.write("\n" + "="*80 + "\n")  # 可选：每次加个分隔符方便查看





if __name__ == "__main__":
    main()
