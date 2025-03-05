def get_severity_label(speaker):
    """根据说话者ID获取严重程度标签（数值类型）"""
    if speaker in ['F01', 'M01', 'M02']:
        return 0  # 轻度
    elif speaker in ['F03', 'M03', 'M04']:
        return 1  # 中度
    elif speaker in ['F04', 'M05']:
        return 2  # 重度
    else:
        return -1  # 未知
