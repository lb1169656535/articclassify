def get_severity_label(speaker):
    """根据说话人 ID 分配严重程度标签"""
    if speaker in ['F03', 'F04', 'M03']:
        return 'VERY LOW'
    elif speaker in ['F01', 'M05']:
        return 'LOW'
    elif speaker in ['M01', 'M02', 'M04']:
        return 'MEDIUM'
    else:
        return 'UNKNOWN'
