import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. Load existing customer dataset
# -----------------------------
customer_data = pd.read_excel("customer_data.xlsx")

# -----------------------------
# 2. Feature-based churn probability
# -----------------------------
# Short tenure, low usage, or many support tickets increase churn risk
customer_data['churn_probability'] = (
    0.5 * (customer_data['tenure_months'] < 12).astype(int) +
    0.3 * (customer_data['monthly_usage'] < 150).astype(int) +
    0.2 * (customer_data['num_support_tickets'] > 3).astype(int)
)

# Simulate actual churn outcome based on probability
customer_data['churned'] = np.random.binomial(1, customer_data['churn_probability'])

# -----------------------------
# 3. Features for churn prediction
# -----------------------------
X = customer_data[['tenure_months', 'monthly_usage', 'num_support_tickets']]
y = customer_data['churned']

# Train Random Forest Classifier on entire dataset (predict at start)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict churn probability and label for all customers
customer_data['predicted_churn_probability'] = clf.predict_proba(X)[:, 1]
customer_data['predicted_churn'] = clf.predict(X)

# Identify high-risk customers immediately
high_risk_customers = customer_data[customer_data['predicted_churn_probability'] > 0.7]

print("High-risk customers at start of analysis:")
print(high_risk_customers.head())

# -----------------------------
# 4. Movie or music products and user ratings
# -----------------------------
products = [
    'The Silent Sea', 'Moonlight Sonata', 'Interstellar', 'Inception',
    'Parasite', 'The Dark Knight', 'Bohemian Rhapsody', 'Avengers: Endgame',
    'La La Land', 'The Godfather'
]

users = ['User_' + str(i) for i in range(1, 11)]
ratings = pd.DataFrame(np.random.randint(1, 6, size=(10, 10)), index=users, columns=products)

# -----------------------------
# 5. Item-item similarity for recommendations
# -----------------------------
item_similarity = cosine_similarity(ratings.T)
item_similarity_df = pd.DataFrame(item_similarity, index=products, columns=products)

# -----------------------------
# 6. Recommendation function
# -----------------------------
def recommend_items(user_id, ratings, item_similarity_df, top_n=3):
    user_ratings = ratings.loc[user_id]
    recommendations = {}
    for item in products:
        if user_ratings[item] < 5:  # Only recommend items not rated 5
            sim_scores = item_similarity_df[item]
            weighted_score = sum(sim_scores * user_ratings) / sum(sim_scores)
            recommendations[item] = weighted_score
    top_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_recs

# -----------------------------
# 7. Generate recommendations for all users
# -----------------------------
all_recommendations = []
for user in users:
    recs = recommend_items(user, ratings, item_similarity_df)
    for product, score in recs:
        all_recommendations.append({'user_id': user, 'recommended_product': product, 'score': score})

recs_df = pd.DataFrame(all_recommendations)

# -----------------------------
# 8. A/B Testing for high-risk customers
# -----------------------------
# Only consider high-risk customers
high_risk_customers = high_risk_customers.copy()

# Randomly assign A/B groups
np.random.seed(42)
high_risk_customers['group'] = np.random.choice(['A', 'B'], size=len(high_risk_customers))

# Simulate intervention effect (Group A gets recommendations, reducing churn by 20%)
high_risk_customers['adjusted_churn_probability'] = high_risk_customers['predicted_churn_probability']
high_risk_customers.loc[high_risk_customers['group'] == 'A', 'adjusted_churn_probability'] *= 0.8

# Simulate churn outcome after intervention
high_risk_customers['churn_after_intervention'] = np.random.binomial(
    1, high_risk_customers['adjusted_churn_probability']
)

# Evaluate A/B results
churn_rate_A = high_risk_customers[high_risk_customers['group'] == 'A']['churn_after_intervention'].mean()
churn_rate_B = high_risk_customers[high_risk_customers['group'] == 'B']['churn_after_intervention'].mean()

print(f"\nA/B Test Results:\nGroup A churn rate: {churn_rate_A:.2f}\nGroup B churn rate: {churn_rate_B:.2f}")

# -----------------------------
# 9. Export data
# -----------------------------
customer_data.to_csv("customer_churn_predictions.csv", index=False)
recs_df.to_csv("customer_recommendations.csv", index=False)
high_risk_customers.to_csv("high_risk_customers_ab_test.csv", index=False)

print("\nPipeline complete. CSV files ready for dashboard integration.")

# -----------------------------
# 10. Example Visuals
# -----------------------------

# Churn counts
churn_counts = customer_data['predicted_churn'].value_counts()
labels = ['Stay', 'Churn']

# Pie chart
plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50','#FF5722'])
plt.title('Predicted Churn Distribution')
plt.savefig("churn_distribution.png")
plt.show()

# A/B churn rates
ab_rates = [churn_rate_A, churn_rate_B]
groups = ['Group A (Intervention)', 'Group B (Control)']

# Bar chart
plt.figure(figsize=(6,4))
plt.bar(groups, ab_rates, color=['#2196F3', '#9E9E9E'])
plt.ylabel('Churn Rate')
plt.title('A/B Test Results: Churn Rate by Group')
plt.ylim(0,1)
for i, v in enumerate(ab_rates):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.savefig("ab_test_results.png")
plt.show()
