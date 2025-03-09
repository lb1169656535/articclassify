import os
from utils.visualization import (
    plot_model_comparison,
    plot_metrics,
    plot_confusion_matrices
)

def main():
  
    try:
        # 模型准确率对比
        results_csv = os.path.abspath('../../results/classification_results.csv')
        plot_model_comparison(
            csv_path=results_csv,
            output_path='../../results/model_comparison.png'
        )
        # 详细指标可视化
        plot_metrics('../../results/detailed_metrics.csv')
        
        # 混淆矩阵可视化
        plot_confusion_matrices('../../results')
    
    except Exception as e:
        pass
    
if __name__ == "__main__":
    main()