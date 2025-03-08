import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(csv_path, output_path='../results/model_comparison.png'):
    """
    生成模型准确率对比可视化图表
    参数：
        csv_path: 结果CSV文件路径
        output_path: 输出图片路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置可视化风格
    plt.figure(figsize=(12, 8), dpi=300)
    
    # 创建柱状图（修改部分开始）
    ax = sns.barplot(
        x='Model', 
        y='Accuracy', 
        hue='Model',  # 新增hue参数
        data=df,
        palette='viridis',
        saturation=0.8,
        legend=False  # 禁用图例显示
    )
    # 修改部分结束
    
    # 设置图表元素
    plt.title('Model Performance Comparison', fontsize=16, pad=20)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 1)

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2%}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', 
            va='center', 
            fontsize=10,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    # 自动调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# 测试用    
# try:
#     results_csv = os.path.abspath('../../results/classification_results.csv')
#     plot_model_comparison(
#         csv_path=results_csv,
#         output_path='../../results/model_comparison.png'
#     )
     
# except Exception as e:
#     pass