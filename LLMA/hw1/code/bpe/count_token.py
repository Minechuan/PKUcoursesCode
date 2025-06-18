def count_whitespaces(file_path):
    """统计文件中的空白字符数量（空格、制表符、换行符等）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            token_num = sum(1 for char in content if char==',')
            return token_num+1
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        return -1
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return -1

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方法：python count_spaces.py 文件名")
        sys.exit(1)

    file_path = sys.argv[1]
    count = count_whitespaces(file_path)
    
    if count >= 0:
        print(f"文件 '{file_path}' 中的,token总数：{count}")