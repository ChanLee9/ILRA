from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 假设有 y_true 和 y_pred
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# 计算分类准确度
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# 计算精确度
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# 计算召回率
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")

# 计算 F1 分数
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")

print(classification_report(y_true, y_pred))
