# Prior probability that a message is spam
p_spam = 0.20

# Probability of seeing FREE when the message is spam
p_free_given_spam = 0.75

# Overall probability of seeing FREE
p_free = 0.20

# Apply Bayes' theorem
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

# Print the probability as a decimal
print("P(Spam | FREE):", p_spam_given_free)

# Convert the probability into percentage
print("Percentage:", p_spam_given_free * 100)
# Show probability rounded to 2 decimal places
print("P(Spam | FREE):", round(p_spam_given_free, 2))

# Show percentage rounded to 2 decimal places
print("Percentage:", round(p_spam_given_free * 100, 2))