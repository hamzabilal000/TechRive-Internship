"""
A small knowledge base for the RAG pipeline.

These are short "documentation" snippets about a fictional SaaS product,
TechRive. In a real system these would come from files, a database, or a
vector store; keeping them inline makes the RAG demo fully offline and
reproducible.
"""

DOCUMENTS: list[str] = [
    "TechRive offers three plans: Free, Pro, and Enterprise. The Free plan "
    "includes up to 3 projects and community support.",
    "The Pro plan costs 15 dollars per user per month and includes unlimited "
    "projects, priority email support, and advanced analytics.",
    "The Enterprise plan adds single sign-on (SSO), a dedicated account "
    "manager, and a 99.9 percent uptime service level agreement.",
    "To reset your password, click 'Forgot password' on the login page and "
    "follow the link sent to your registered email address.",
    "You can export all of your data at any time from Settings > Data Export "
    "as a downloadable ZIP archive of CSV and JSON files.",
    "TechRive stores data encrypted at rest using AES-256 and encrypts all "
    "network traffic in transit using TLS 1.3.",
    "Customer support is available Monday to Friday, 9am to 6pm UTC, via email "
    "and live chat for Pro and Enterprise customers.",
    "The mobile app is available for both iOS and Android and supports offline "
    "editing that syncs automatically when you reconnect.",
    "Billing is processed monthly. You can update your payment method or "
    "billing address under Settings > Billing at any time.",
    "TechRive integrates with Slack, GitHub, Google Drive, and Zapier. New "
    "integrations are added regularly based on customer requests.",
]
