import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_and_save_csv(csv_file_path, output_dir):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path, header=None, names=['X', 'Y'])

    # 绘制图形
    plt.plot(df['X'], df['Y'], marker='o')
    plt.title(f'Data Plot from {os.path.basename(csv_file_path)}')
    plt.xlabel('iteration')
    plt.ylabel('uncoloring')
    plt.grid(True)

    # 构建输出文件名（更改扩展名为 .png）
    output_file = os.path.splitext(os.path.basename(csv_file_path))[0] + ".png"
    output_path = os.path.join(output_dir, output_file)

    # 保存图形为 PNG 文件
    plt.savefig(output_path)

    # 清除当前图形
    plt.clf()

def main():
    # 设置要遍历的目录
    directory = os.getcwd()  # 当前目录，可修改为任意目录路径

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 CSV 文件
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(directory, filename)
            plot_and_save_csv(csv_file_path, directory)

if __name__ == "__main__":
    main()