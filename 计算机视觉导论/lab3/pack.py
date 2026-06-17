import os
import zipfile


SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "earthmover","img_outputs"}
SKIP_SUFFIXES = {".pyc", ".pth"}


def zipHW3(input_path: str, output_path: str, zip_name: str):
    output_abs = os.path.abspath(output_path)
    root_prefix = f"HW3_{zip_name}"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path, dirnames, filenames in os.walk(input_path):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in SKIP_DIRS
            ]

            fpath = path.replace(input_path, root_prefix)
            for filename in filenames:
                src_path = os.path.join(path, filename)
                if os.path.abspath(src_path) == output_abs:
                    continue
                if filename.endswith(".zip"):
                    continue
                if any(filename.endswith(suffix) for suffix in SKIP_SUFFIXES):
                    continue
                zip_file.write(src_path, os.path.join(fpath, filename))


if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 请用你的学号和姓名替换下面的内容，注意参照例子的格式，使用拼音而非中文
    id = 2300013218
    name = 'MaoChuan'
    # ---------------------------------------------------------

    zip_name = f'{id}_{name}.zip'
    input_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), zip_name)
 
    zipHW3(input_path, output_path, zip_name.split(".")[0])
