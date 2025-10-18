import pandas as pd
gr={}
lis=set()
f=r'drop_col_combined.csv'
data=pd.read_csv(f,engine='python')
c=data.groupby('label').size()
for l,cn in c.items():
    if l in gr:
        gr[l]+=cn
    else:
        gr[l]=cn
print(gr)
{0.0: 22852, 1.0: 69531, 2.0: 70435, 3.0: 18219, 4.0: 11704, 5.0: 10952}

# imp = pd.DataFrame({'Feature': train_feat.columns, 'Importance': rf.feature_importances_})
# imp = imp.sort_values(by='Importance', ascending=False)

# 
# # Extract sorted features and scores
# sorted_features = imp['Feature'].tolist()
# sorted_scores = imp['Importance'].tolist()
# # Plot feature importances
# plt.figure(figsize=(10, 8))
# plt.barh(sorted_features, sorted_scores, color='skyblue')
# plt.xlabel('F score')
# plt.title('Feature Importance')
# plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
# plt.show()
