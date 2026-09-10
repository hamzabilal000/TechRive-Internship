"""
Week 2 - Task 3: NLP text classifier (TF-IDF + Logistic Regression).

A small, self-contained labelled text set of short customer messages is
classified into three support categories: "billing", "technical", "general".
This keeps the task fully offline and deterministic (no dataset download).
The same TF-IDF + LogReg approach is the classic strong baseline for text
classification, and it foreshadows the support-triage theme of the capstone.

Run:
    python 03_nlp_tfidf.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# --------------------------------------------------------------------------- #
# Small labelled dataset (message, category)
# --------------------------------------------------------------------------- #
DATA = [
    # billing
    ("I was charged twice for my subscription this month", "billing"),
    ("Why is my invoice higher than last month", "billing"),
    ("I want a refund for the duplicate payment", "billing"),
    ("My credit card was billed but I cancelled already", "billing"),
    ("Can you explain these extra charges on my bill", "billing"),
    ("Please update my payment method to a new card", "billing"),
    ("I need a copy of my receipt for accounting", "billing"),
    ("The discount code did not apply to my order total", "billing"),
    ("You overcharged me and I want the difference back", "billing"),
    ("How do I change my billing address", "billing"),
    ("My subscription renewed but I meant to cancel it", "billing"),
    ("The invoice total does not match what I was quoted", "billing"),
    ("I was billed in the wrong currency", "billing"),
    ("Can I get a refund for the unused portion of my plan", "billing"),
    ("There is a mysterious charge from your company on my statement", "billing"),
    ("My payment failed but money still left my account", "billing"),
    ("Please cancel my subscription and stop charging my card", "billing"),
    ("I need an itemised breakdown of last month's charges", "billing"),
    # technical
    ("The app crashes every time I open the dashboard", "technical"),
    ("I cannot log in, it says invalid password", "technical"),
    ("The page keeps loading forever and never finishes", "technical"),
    ("Getting a 500 error when I upload a file", "technical"),
    ("The sync feature stopped working after the update", "technical"),
    ("My data is not saving when I click submit", "technical"),
    ("The mobile app freezes on the login screen", "technical"),
    ("API requests are timing out with a connection error", "technical"),
    ("The export button does nothing when clicked", "technical"),
    ("Notifications are not being delivered to my phone", "technical"),
    ("The search bar returns no results even for valid queries", "technical"),
    ("I keep getting logged out every few minutes", "technical"),
    ("The dashboard charts fail to render and show a blank box", "technical"),
    ("File uploads fail with a network timeout error", "technical"),
    ("The app shows a white screen after the latest update", "technical"),
    ("Two-factor authentication codes never arrive", "technical"),
    ("The integration with Slack broke and throws an error", "technical"),
    ("Clicking save produces a server error and loses my work", "technical"),
    ("Video playback stutters and buffers constantly", "technical"),
    # general
    ("What are your customer support hours", "general"),
    ("Do you offer a student discount plan", "general"),
    ("How do I upgrade to the premium tier", "general"),
    ("Where can I find your documentation", "general"),
    ("Can I invite team members to my workspace", "general"),
    ("Is there a mobile version of the product", "general"),
    ("How do I delete my account permanently", "general"),
    ("What integrations do you support", "general"),
    ("Do you have a free trial available", "general"),
    ("How can I contact the sales team", "general"),
    ("What is the difference between the basic and pro plans", "general"),
    ("Do you have an app for tablets", "general"),
    ("Can I change my username after signing up", "general"),
    ("Where do I download the desktop client", "general"),
    ("Is my data backed up automatically", "general"),
    ("Do you support single sign on for organisations", "general"),
    ("How many projects can I create on the free plan", "general"),
    ("What languages is your product available in", "general"),
    ("Can I schedule a demo of the product", "general"),
    ("How do I turn on dark mode", "general"),
]


def main() -> None:
    texts = [t for t, _ in DATA]
    labels = [c for _, c in DATA]
    print(f"Dataset: {len(texts)} messages across "
          f"{len(set(labels))} categories {sorted(set(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, preds):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Try the trained model on brand-new messages.
    demo = [
        "I think you double billed my account again",
        "The website throws an error when I submit the form",
        "Do you have a plan for nonprofits",
    ]
    print("Predictions on new messages:")
    for msg, pred in zip(demo, model.predict(demo)):
        print(f"  [{pred:<9}] {msg}")


if __name__ == "__main__":
    main()
